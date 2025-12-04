#!/usr/bin/env python3
"""
Canny Walls + Simple Robot Sprite + 5-beam LIDAR
— Full script with wall-safe minimap (max-pool downsample), convexified/simplified
  planning, and planning checks against the full-resolution wall mask so plans
  do not pass through walls that vanish under naive downsampling.

Key changes vs prior:
 - minimap walls produced by **max-pooling** (preserves thin walls)
 - minimap click -> world conversion uses minimap size but minimap visuals reflect pooled walls
 - planner builds graph from full-res wall mask and verifies line-of-sight using full-res mask
 - path simplification (shortcutting) uses full-res LOS checks
 - LIDAR beams shorter and thinner
"""
import argparse
from pathlib import Path
import sys
import math
from typing import Dict, List, Tuple
import pymunk

import cv2
import numpy as np
import pygame

# optional planning libs
try:
    import networkx as nx
except Exception:
    nx = None
try:
    import pulp
except Exception:
    pulp = None

class StaticObject:
    """Base class for any static object placed in the map."""
    def __init__(self, space, color=(180, 180, 180)):
        self.space = space
        self.color = color
        self.body = space.static_body   # all static objects share the static body
        self.shape = None

    def render(self, surface, camera):
        """Override per shape."""
        pass

    def render_minimap(self, surface, scale):
        """Override per shape."""
        pass
class StaticRectangle(StaticObject):
    def __init__(self, space, x, y, width, height, angle=0.0, color=(180, 150, 80)):
        super().__init__(space, color)
        
        # Create Pymunk Poly
        rect = pymunk.Poly.create_box(self.body, (width, height))
        rect.body.position = (x, y)
        rect.body.angle = angle
        rect.elasticity = 0.1
        rect.friction = 0.9
        
        self.shape = rect
        space.add(rect)

        self.w = width
        self.h = height
        self.angle = angle

    def render(self, surface, camera):
        x, y = self.shape.body.position
        sx, sy = camera.world_to_screen((x, y))
        w = self.w * camera.zoom
        h = self.h * camera.zoom

        rect = pygame.Surface((w, h), pygame.SRCALPHA)
        rect.fill(self.color)
        rect = pygame.transform.rotate(rect, -math.degrees(self.shape.body.angle))

        surface.blit(rect, (sx - rect.get_width() // 2,
                            sy - rect.get_height() // 2))

    def render_minimap(self, surface, scale):
        x, y = self.shape.body.position
        w = self.w // scale
        h = self.h // scale
        pygame.draw.rect(surface, self.color,
                         pygame.Rect(int(x // scale), int(y // scale), w, h))
        
class StaticCircle(StaticObject):
    def __init__(self, space, x, y, radius, color=(200, 80, 80)):
        super().__init__(space, color)

        circle = pymunk.Circle(self.body, radius)
        circle.body.position = (x, y)
        circle.elasticity = 0.1
        circle.friction = 0.9
        
        self.shape = circle
        space.add(circle)

        self.radius = radius

    def render(self, surface, camera):
        x, y = self.shape.body.position
        sx, sy = camera.world_to_screen((x, y))
        r = int(self.radius * camera.zoom)
        pygame.draw.circle(surface, self.color, (int(sx), int(sy)), r)

    def render_minimap(self, surface, scale):
        x, y = self.shape.body.position
        r = self.radius // scale
        pygame.draw.circle(surface, self.color,
                           (int(x // scale), int(y // scale)), int(r))


# ------------------------ utilities ------------------------
def load_and_prepare_mask(image_path: Path, world_w: int, world_h: int, low: int, high: int):
    """
    Load image, compute Canny edges, dilate to produce walls binary (walls_bin),
    create pygame surfaces/masks for full-resolution display and collision checking.
    Returns: wall_mask (pygame.Mask), wall_surf (RGBA surface for overlay), display_surf (RGB)
    and walls_bin (uint8 0/255)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    img = cv2.resize(img, (world_w, world_h))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(gray, low, high)

    kernel = np.ones((3, 3), np.uint8)
    walls = cv2.dilate(edges, kernel, iterations=2)

    h, w = walls.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = walls
    rgba[:, :, 1] = walls
    rgba[:, :, 2] = walls
    rgba[:, :, 3] = walls
    wall_surf = pygame.image.frombuffer(rgba.tobytes(), (w, h), 'RGBA')
    wall_mask = pygame.mask.from_surface(wall_surf)

    # displayable background (dimmed original + green walls highlight)
    color_edges = np.zeros_like(img)
    color_edges[:, :, 1] = walls
    dimmed = (img * 0.4).astype(np.uint8)
    display = cv2.add(dimmed, color_edges)
    display_surf = pygame.image.frombuffer(cv2.cvtColor(display, cv2.COLOR_BGR2RGB).tobytes(), (w, h), 'RGB')

    walls_bin = (walls > 0).astype(np.uint8) * 255
    return wall_mask, wall_surf, display_surf, walls_bin

def downsample_occupancy_maxpool(walls_bin: np.ndarray, minimap_w: int, minimap_h: int) -> np.ndarray:
    """
    Downsample binary occupancy (walls_bin 0/255) using max-pooling so thin walls are preserved.
    Returns a minimap-sized binary image (0/255).
    """
    world_h, world_w = walls_bin.shape
    # Choose integer pooling factors so output fits nicely
    # We compute scale factors (pool sizes) to cover world -> minimap
    pool_x = max(1, math.ceil(world_w / minimap_w))
    pool_y = max(1, math.ceil(world_h / minimap_h))
    pool = max(pool_x, pool_y)  # use square blocks for simplicity

    # Dilate with a pool x pool kernel to propagate walls within each block, then sample
    kernel = np.ones((pool, pool), np.uint8)
    pooled = cv2.dilate(walls_bin, kernel, iterations=1)

    # Now sample pooled into minimap size using nearest/area sampling which won't blur walls
    mini = cv2.resize(pooled, (minimap_w, minimap_h), interpolation=cv2.INTER_NEAREST)
    # ensure binary 0/255
    mini = (mini > 0).astype(np.uint8) * 255
    return mini

def make_minimap_display(bg_display: np.ndarray, walls_bin: np.ndarray, minimap_w: int, minimap_h: int):
    """
    Create a minimap RGB surface for display:
     - downsample the full-res background image (bg_display) to minimap using INTER_AREA
     - compute pooled walls via maxpool and overlay green highlights
    Returns pygame.Surface RGB (minimap_w x minimap_h) and pooled walls binary (minimap-sized)
    """
    # bg_display is RGB numpy (H,W,3)
    mini_bg = cv2.resize(cv2.cvtColor(bg_display, cv2.COLOR_BGR2RGB), (minimap_w, minimap_h), interpolation=cv2.INTER_AREA)
    pooled = downsample_occupancy_maxpool(walls_bin, minimap_w, minimap_h)
    # overlay green walls
    overlay = mini_bg.copy()
    mask_idx = pooled > 0
    overlay[mask_idx, 0] = overlay[mask_idx, 0] * 0.4  # dim red
    overlay[mask_idx, 1] = 255  # highlight green
    surf = pygame.image.frombuffer(overlay.tobytes(), (minimap_w, minimap_h), 'RGB')
    return surf, pooled

# ------------------------ grid graph & planning ------------------------
def build_grid_graph_from_mask(wall_mask: pygame.mask.Mask,
                               world_w: int, world_h: int,
                               grid_step: int = 16) -> Tuple[Dict[Tuple[int, int], int], List[Tuple[int, int, float]]]:
    """
    Build grid nodes sampled on full-resolution world (using wall_mask to skip nodes inside walls).
    """
    node_id: Dict[Tuple[int, int], int] = {}
    id_counter = 0
    for y in range(grid_step // 2, world_h, grid_step):
        for x in range(grid_step // 2, world_w, grid_step):
            px = min(world_w - 1, x)
            py = min(world_h - 1, y)
            if wall_mask.get_at((px, py)) == 0:
                node_id[(px, py)] = id_counter
                id_counter += 1

    edges = []
    offsets = [(grid_step, 0), (0, grid_step), (-grid_step, 0), (0, -grid_step),
               (grid_step, grid_step), (grid_step, -grid_step), (-grid_step, grid_step), (-grid_step, -grid_step)]
    for (x, y), uid in list(node_id.items()):
        for dx, dy in offsets:
            nx_coord = (x + dx, y + dy)
            if nx_coord in node_id:
                vid = node_id[nx_coord]
                w = math.hypot(dx, dy)
                edges.append((uid, vid, w))
    return node_id, edges

def find_nearest_node(coord_to_id: Dict[Tuple[int, int], int], point: Tuple[float, float]) -> Tuple[Tuple[int, int], int]:
    best = None
    best_id = None
    best_d = float('inf')
    px, py = point
    for (x, y), nid in coord_to_id.items():
        d = (x - px) ** 2 + (y - py) ** 2
        if d < best_d:
            best_d = d
            best = (x, y)
            best_id = nid
    return best, best_id

def solve_shortest_path_ilp(coord_to_id: Dict[Tuple[int, int], int],
                            edges: List[Tuple[int, int, float]],
                            start_id: int, goal_id: int) -> List[int]:
    if pulp is None:
        return []
    prob = pulp.LpProblem('shortest_path', pulp.LpMinimize)
    x_vars = {}
    for i, (u, v, w) in enumerate(edges):
        x_vars[i] = pulp.LpVariable(f'x_{i}', cat='Binary')
    prob += pulp.lpSum([w * x_vars[i] for i, (_, _, w) in enumerate(edges)])

    node_count = len(coord_to_id)
    node_in = {nid: [] for nid in range(node_count)}
    node_out = {nid: [] for nid in range(node_count)}
    for i, (u, v, w) in enumerate(edges):
        node_out[u].append(i)
        node_in[v].append(i)

    for nid in range(node_count):
        if nid == start_id:
            prob += pulp.lpSum([x_vars[i] for i in node_out[nid]]) == 1
        elif nid == goal_id:
            prob += pulp.lpSum([x_vars[i] for i in node_in[nid]]) == 1
        else:
            prob += pulp.lpSum([x_vars[i] for i in node_in[nid]]) - pulp.lpSum([x_vars[i] for i in node_out[nid]]) == 0

    solver = pulp.PULP_CBC_CMD(msg=False)
    res = prob.solve(solver)
    if res != 1:
        return []

    selected = [i for i in x_vars if pulp.value(x_vars[i]) >= 0.5]
    adj = {nid: [] for nid in range(node_count)}
    for i in selected:
        u, v, w = edges[i]
        adj[u].append(v)

    path = [start_id]
    cur = start_id
    visited = set([cur])
    while cur != goal_id:
        if not adj.get(cur):
            break
        nxt = adj[cur][0]
        if nxt in visited:
            break
        path.append(nxt)
        visited.add(nxt)
        cur = nxt
    if cur != goal_id:
        return []
    return path

def solve_shortest_path_graph(coord_to_id: Dict[Tuple[int, int], int],
                              edges: List[Tuple[int, int, float]],
                              start_id: int, goal_id: int) -> List[int]:
    if nx is None:
        return []
    G = nx.DiGraph()
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    try:
        node_path = nx.shortest_path(G, source=start_id, target=goal_id, weight='weight')
        return node_path
    except Exception:
        return []

# ------------------------ geometry helpers ------------------------
def line_free_between_points(p1: Tuple[float, float], p2: Tuple[float, float],
                             wall_mask: pygame.mask.Mask, world_w: int, world_h: int) -> bool:
    """
    Conservative raster-sampled line-of-sight test against full-res wall_mask.
    """
    x1, y1 = p1
    x2, y2 = p2
    steps = max(int(math.hypot(x2 - x1, y2 - y1)), 1)
    for i in range(0, steps + 1):
        t = i / steps
        ix = int(round(x1 + (x2 - x1) * t))
        iy = int(round(y1 + (y2 - y1) * t))
        if ix < 0 or iy < 0 or ix >= world_w or iy >= world_h:
            return False
        if wall_mask.get_at((ix, iy)):
            return False
    return True

def convexify_points(points: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
    pts = [(float(x), float(y)) for (x, y) in points]
    pts = sorted(set(pts))
    if len(pts) <= 2:
        return [(int(round(x)), int(round(y))) for x, y in pts]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return [(int(round(x)), int(round(y))) for x, y in hull]

def shortcut_and_simplify(path_coords: List[Tuple[int, int]],
                          wall_mask: pygame.mask.Mask, world_w: int, world_h: int) -> List[Tuple[int, int]]:
    if not path_coords:
        return []
    simplified = [path_coords[0]]
    i = 0
    n = len(path_coords)
    while i < n - 1:
        far = i + 1
        for j in range(n - 1, i, -1):
            if line_free_between_points(path_coords[i], path_coords[j], wall_mask, world_w, world_h):
                far = j
                break
        simplified.append(path_coords[far])
        i = far
    out = []
    for p in simplified:
        if not out or out[-1] != p:
            out.append(p)
    return out

# ------------------------ LIDAR ------------------------
def get_lidar_distances(x: float, y: float, heading: float, mask: pygame.mask.Mask,
                        world_size: Tuple[int, int], num_beams: int = 5, max_range: int = 120) -> List[int]:
    w, h = world_size
    half_span = math.pi / 2.0
    distances = []
    if num_beams == 1:
        rel_angles = [0.0]
    else:
        rel_angles = [(-half_span) + i * (2 * half_span) / (num_beams - 1) for i in range(num_beams)]

    for rel in rel_angles:
        angle = heading + rel
        dx = math.cos(angle)
        dy = math.sin(angle)
        dist_hit = max_range
        for d in range(1, max_range + 1):
            px = int(round(x + dx * d))
            py = int(round(y + dy * d))
            if not (0 <= px < w and 0 <= py < h):
                dist_hit = d - 1
                break
            if mask.get_at((px, py)):
                dist_hit = d - 1
                break
        distances.append(dist_hit)
    return distances

# ------------------------ main simulation ------------------------
def run(image_path: Path,
        world_w: int = 1920, world_h: int = 1080,
        win_w: int = 800, win_h: int = 600,
        zoom: int = 10,
        low: int = 50, high: int = 150):
    pygame.init()
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption('Canny Walls - Wall-safe Minimap + Convex Planning')
    clock = pygame.time.Clock()

    # full-res wall mask and display image + binary
    wall_mask, wall_surf, bg_surf, walls_bin = load_and_prepare_mask(image_path, world_w, world_h, low, high)

    # small robot sprite
    player_w = 4
    player_h = 4
    player_x = world_w // 2
    player_y = world_h // 2
    speed = 3

    player_surf = pygame.Surface((player_w, player_h), flags=pygame.SRCALPHA)
    player_surf.fill((200, 50, 50))
    if player_w >= 3 and player_h >= 3:
        player_surf.set_at((player_w - 1, player_h // 2), (255, 200, 50))
    player_mask = pygame.mask.from_surface(player_surf)

    # lidar params (smaller beams)
    num_lidar = 5
    lidar_max = 120
    heading = 0.0

    # planning state
    path_waypoints: List[Tuple[int, int]] = []
    path_index = 0
    graph_cache = None  # (coord_to_id, edges)

    show_walls = True
    running = True

    try:
        font = pygame.font.SysFont(None, 20)
    except Exception:
        pygame.font.init()
        font = pygame.font.SysFont(None, 20)

    # minimap params (visual)
    minimap_w = min(240, win_w // 4)
    minimap_h = min(160, win_h // 4)
    minimap_rect = pygame.Rect(win_w - minimap_w - 8, 8, minimap_w, minimap_h)
    # create minimap surfaces (we'll regenerate each frame to reflect latest walls_bin/bg)
    factor_x = minimap_w / float(world_w)
    factor_y = minimap_h / float(world_h)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    show_walls = not show_walls
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                # convert click to world coords via minimap mapping (minimap shows pooled walls so user sees preserved walls)
                if minimap_rect.collidepoint((mx, my)):
                    # world coords mapped proportionally
                    wx = int((mx - minimap_rect.x) / minimap_w * world_w)
                    wy = int((my - minimap_rect.y) / minimap_h * world_h)

                    # build graph from full-res wall_mask (ensures obstacles in full-res are considered)
                    if graph_cache is None:
                        coord_to_id, edges = build_grid_graph_from_mask(wall_mask, world_w, world_h, grid_step=16)
                        graph_cache = (coord_to_id, edges)
                    else:
                        coord_to_id, edges = graph_cache

                    start_coord, start_id = find_nearest_node(coord_to_id, (player_x, player_y))
                    goal_coord, goal_id = find_nearest_node(coord_to_id, (wx, wy))

                    planned_ids: List[int] = []
                    if pulp is not None and start_id is not None and goal_id is not None:
                        planned_ids = solve_shortest_path_ilp(coord_to_id, edges, start_id, goal_id)
                    if not planned_ids and nx is not None and start_id is not None and goal_id is not None:
                        planned_ids = solve_shortest_path_graph(coord_to_id, edges, start_id, goal_id)

                    if planned_ids:
                        id_to_coord = {nid: coord for coord, nid in coord_to_id.items()}
                        waypoints = [id_to_coord[nid] for nid in planned_ids]

                        # simplify via LOS on full-res mask
                        simplified = shortcut_and_simplify(waypoints, wall_mask, world_w, world_h)

                        # attempt convex hull of simplified points but only accept if hull edges are free on full-res mask
                        hull = convexify_points(simplified)
                        hull_ok = True
                        if len(hull) >= 2:
                            for i in range(len(hull)):
                                a = hull[i]
                                b = hull[(i + 1) % len(hull)]
                                if not line_free_between_points(a, b, wall_mask, world_w, world_h):
                                    hull_ok = False
                                    break
                        else:
                            hull_ok = False

                        if hull_ok and len(hull) >= 2:
                            path_waypoints = hull
                            path_index = 0
                        else:
                            path_waypoints = simplified
                            path_index = 0

        # manual input
        keys = pygame.key.get_pressed()
        move_dx = 0
        move_dy = 0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            move_dx -= speed
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            move_dx += speed
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            move_dy -= speed
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            move_dy += speed

        if (move_dx != 0 or move_dy != 0):
            path_waypoints = []
            path_index = 0

        tentative_x = player_x + move_dx
        tentative_y = player_y + move_dy
        tentative_x = max(0, min(world_w - player_w, tentative_x))
        tentative_y = max(0, min(world_h - player_h, tentative_y))

        overlap = wall_mask.overlap(player_mask, (int(tentative_x), int(tentative_y)))
        if overlap is None:
            if move_dx != 0 or move_dy != 0:
                heading = math.atan2(move_dy, move_dx) if (move_dx != 0 or move_dy != 0) else heading
            player_x = tentative_x
            player_y = tentative_y
        else:
            overlap_x = wall_mask.overlap(player_mask, (int(tentative_x), int(player_y)))
            overlap_y = wall_mask.overlap(player_mask, (int(player_x), int(tentative_y)))
            if overlap_x is None:
                player_x = tentative_x
                if move_dx != 0:
                    heading = math.atan2(0, move_dx)
            if overlap_y is None:
                player_y = tentative_y
                if move_dy != 0:
                    heading = math.atan2(move_dy, 0)

        # path following
        if path_waypoints and path_index < len(path_waypoints):
            tx, ty = path_waypoints[path_index]
            vx = tx - player_x
            vy = ty - player_y
            dist = math.hypot(vx, vy)
            if dist < 2.0:
                path_index += 1
            else:
                desired_heading = math.atan2(vy, vx)
                step_x = (vx / dist) * speed
                step_y = (vy / dist) * speed

                world_center_x = player_x + player_w / 2.0
                world_center_y = player_y + player_h / 2.0
                lidar = get_lidar_distances(world_center_x, world_center_y, desired_heading,
                                            wall_mask, (world_w, world_h), num_beams=num_lidar, max_range=lidar_max)
                mid = len(lidar) // 2
                front_slice = lidar[max(0, mid-1): min(len(lidar), mid+2)]
                if min(front_slice) < 8:
                    path_waypoints = []
                    path_index = 0
                else:
                    tentative_x = player_x + step_x
                    tentative_y = player_y + step_y
                    tentative_x = max(0, min(world_w - player_w, tentative_x))
                    tentative_y = max(0, min(world_h - player_h, tentative_y))
                    overlap = wall_mask.overlap(player_mask, (int(tentative_x), int(tentative_y)))
                    if overlap is None:
                        player_x = tentative_x
                        player_y = tentative_y
                        heading = desired_heading
                    else:
                        overlap_x = wall_mask.overlap(player_mask, (int(tentative_x), int(player_y)))
                        overlap_y = wall_mask.overlap(player_mask, (int(player_x), int(tentative_y)))
                        if overlap_x is None:
                            player_x = tentative_x
                        if overlap_y is None:
                            player_y = tentative_y
                        if overlap_x is not None and overlap_y is not None:
                            path_waypoints = []
                            path_index = 0

        # camera crop
        crop_w = max(1, win_w // zoom)
        crop_h = max(1, win_h // zoom)
        player_cx = player_x + player_w // 2
        player_cy = player_y + player_h // 2
        cam_x = int(player_cx - crop_w // 2)
        cam_y = int(player_cy - crop_h // 2)
        cam_x = max(0, min(world_w - crop_w, cam_x))
        cam_y = max(0, min(world_h - crop_h, cam_y))
        cam_rect = pygame.Rect(cam_x, cam_y, crop_w, crop_h)

        # subsurfaces
        try:
            bg_crop = bg_surf.subsurface(cam_rect)
            wall_crop = wall_surf.subsurface(cam_rect) if show_walls else None
        except ValueError:
            bg_crop = pygame.Surface((crop_w, crop_h))
            bg_crop.blit(bg_surf, (0, 0), cam_rect)
            if show_walls:
                wall_crop = pygame.Surface((crop_w, crop_h), flags=pygame.SRCALPHA)
                wall_crop.blit(wall_surf, (0, 0), cam_rect)
            else:
                wall_crop = None

        bg_scaled = pygame.transform.scale(bg_crop, (win_w, win_h))
        wall_scaled = pygame.transform.scale(wall_crop, (win_w, win_h)) if (wall_crop is not None) else None

        # build minimap display from full-res data (preserve walls via maxpool)
        # reuse the precomputed walls_bin and full-res bg (bg_surf) -- convert bg_surf to numpy
        bg_arr = pygame.surfarray.array3d(bg_surf)  # (W,H,3) with RGB order but transposed axes; handle carefully
        # pygame.surfarray.array3d returns (width, height, 3) axis order; transpose to (H,W,3)
        bg_arr = np.transpose(bg_arr, (1, 0, 2)).copy()
        minimap_surf, minimap_pooled = make_minimap_display(bg_arr, walls_bin, minimap_w, minimap_h)

        # draw frame
        screen.fill((0, 0, 0))
        screen.blit(bg_scaled, (0, 0))
        if wall_scaled is not None:
            screen.blit(wall_scaled, (0, 0))

        # draw planned path in main view
        if path_waypoints and len(path_waypoints) >= 2:
            pts = []
            for (x, y) in path_waypoints:
                sx = int((x - cam_x) * zoom)
                sy = int((y - cam_y) * zoom)
                pts.append((sx, sy))
            if len(pts) >= 2:
                pygame.draw.lines(screen, (0, 200, 255), False, pts, 2)

        # draw player
        screen_player_x = int((player_x - cam_x) * zoom)
        screen_player_y = int((player_y - cam_y) * zoom)
        screen_player_w = max(1, int(player_w * zoom))
        screen_player_h = max(1, int(player_h * zoom))
        player_vis = pygame.transform.scale(player_surf, (screen_player_w, screen_player_h))
        screen.blit(player_vis, (screen_player_x, screen_player_y))

        # draw lidar beams
        world_center_x = player_x + player_w / 2.0
        world_center_y = player_y + player_h / 2.0
        lidar_ranges = get_lidar_distances(world_center_x, world_center_y, heading, wall_mask, (world_w, world_h),
                                           num_beams=num_lidar, max_range=lidar_max)
        half_span = math.pi / 2.0
        rel_angles = [(-half_span) + i * (2 * half_span) / (num_lidar - 1) for i in range(num_lidar)]
        for rel, rng in zip(rel_angles, lidar_ranges):
            ang = heading + rel
            ex = world_center_x + math.cos(ang) * rng
            ey = world_center_y + math.sin(ang) * rng
            sx = int((world_center_x - cam_x) * zoom)
            sy = int((world_center_y - cam_y) * zoom)
            exs = int((ex - cam_x) * zoom)
            eys = int((ey - cam_y) * zoom)
            rfrac = max(0.0, min(1.0, rng / float(lidar_max)))
            col = (255, int(150 * rfrac + 50), int(60 * rfrac + 20))
            pygame.draw.line(screen, col, (sx, sy), (exs, eys), 1)
            pygame.draw.circle(screen, col, (exs, eys), 2)

        # minimap render
        screen.blit(minimap_surf, (minimap_rect.x, minimap_rect.y))
        # draw path on minimap (map world -> minimap using proportional scaling)
        if path_waypoints:
            mini_pts = []
            for (x, y) in path_waypoints:
                mx = minimap_rect.x + int(x / world_w * minimap_w)
                my = minimap_rect.y + int(y / world_h * minimap_h)
                mini_pts.append((mx, my))
            if len(mini_pts) >= 2:
                pygame.draw.lines(screen, (0, 255, 255), False, mini_pts, 2)
        # draw robot on minimap
        mini_rx = minimap_rect.x + int(player_x / world_w * minimap_w)
        mini_ry = minimap_rect.y + int(player_y / world_h * minimap_h)
        pygame.draw.circle(screen, (255, 0, 0), (mini_rx, mini_ry), max(2, int(player_w * minimap_w / world_w)))

        # HUD
        mid_idx = len(lidar_ranges) // 2
        mid_range = lidar_ranges[mid_idx] if lidar_ranges else 0
        hud = font.render(f'Zoom: {zoom}x  Pos: ({int(player_x)},{int(player_y)})  LIDAR(mid)={mid_range}', True, (255, 255, 255))
        screen.blit(hud, (8, 8))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

# ------------------------ CLI ------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Canny Walls - Wall-safe Minimap + Convex Planning')
    p.add_argument('--image', type=Path, default=Path('IMG_6705.jpg'), help='Image (default IMG_6705.jpg)')
    p.add_argument('--width', type=int, default=1920, help='World/image width in pixels')
    p.add_argument('--height', type=int, default=1080, help='World/image height in pixels')
    p.add_argument('--win-width', type=int, default=800, help='Window width')
    p.add_argument('--win-height', type=int, default=600, help='Window height')
    p.add_argument('--zoom', type=int, default=10, help='Zoom factor (integer)')
    p.add_argument('--low', type=int, default=50)
    p.add_argument('--high', type=int, default=150)
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if not args.image.exists():
        print(f"Image not found: {args.image}")
        sys.exit(1)
    run(args.image,
        world_w=args.width, world_h=args.height,
        win_w=args.win_width, win_h=args.win_height,
        zoom=args.zoom,
        low=args.low, high=args.high)
