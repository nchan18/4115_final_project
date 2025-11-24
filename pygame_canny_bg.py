#!/usr/bin/env python3
"""
Minimal Pygame simulation: load a single image, compute Canny edges and treat edges as walls.

This variant implements a zoomed viewport that follows the player.

Behavior:
 - Loads `IMG_6705.jpg` by default (or pass --image)
 - Computes Canny edges and dilates them to create thicker walls
 - Creates a pygame mask from the walls and an *inflated* mask for collision/planning
 - A player square (controllable with WASD / arrow keys) cannot pass through wall pixels
 - Camera/viewport follows the player and displays a zoomed-in view
"""
import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
import pygame
import math
from typing import Dict, List, Tuple
from collections import deque

# optional libraries for planning
try:
    import networkx as nx
except Exception:
    nx = None
try:
    import pulp
except Exception:
    pulp = None


def load_and_prepare_mask(image_path: Path, world_w: int, world_h: int, low: int, high: int):
    """
    Returns:
      - wall_mask: pygame.mask.Mask for the *original* edges (visual)
      - wall_surf: pygame.Surface RGBA visual surface for edges
      - display_surf: RGB surface (dimmed original + green edges) for background display
      - walls_bin: numpy uint8 binary array (0/255) of walls used to create inflated mask later
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    img = cv2.resize(img, (world_w, world_h))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(gray, low, high)

    kernel = np.ones((3, 3), np.uint8)
    walls = cv2.dilate(edges, kernel, iterations=2)  # still fairly thin for display

    # create an RGBA surface where wall pixels set alpha=255 (for visual overlay)
    h, w = walls.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = walls
    rgba[:, :, 1] = walls
    rgba[:, :, 2] = walls
    rgba[:, :, 3] = walls

    wall_surf = pygame.image.frombuffer(rgba.tobytes(), (w, h), 'RGBA')
    wall_mask = pygame.mask.from_surface(wall_surf)

    # prepare a displayable background (dimmed original with walls highlighted)
    color_edges = np.zeros_like(img)
    color_edges[:, :, 1] = walls  # green channel highlight
    dimmed = (img * 0.4).astype(np.uint8)
    display = cv2.add(dimmed, color_edges)
    display_surf = pygame.image.frombuffer(cv2.cvtColor(display, cv2.COLOR_BGR2RGB).tobytes(), (w, h), 'RGB')

    # return also the binary walls array for inflation later
    walls_bin = (walls > 0).astype(np.uint8) * 255

    return wall_mask, wall_surf, display_surf, walls_bin


def build_grid_graph_from_mask(wall_mask: pygame.mask.Mask,
                               world_w: int, world_h: int,
                               grid_step: int = 16) -> Tuple[Dict[Tuple[int, int], int], List[Tuple[int, int, float]]]:
    """
    Build grid nodes spaced by grid_step. Return mapping coord->id and list of directed edges (u,v,weight).
    Expects wall_mask to already represent the *inflated* walls (safe clearance).
    """
    node_id: Dict[Tuple[int, int], int] = {}
    id_counter = 0
    # sample grid positions with offset to avoid edge clipping
    for y in range(grid_step // 2, world_h, grid_step):
        for x in range(grid_step // 2, world_w, grid_step):
            px = min(world_w - 1, x)
            py = min(world_h - 1, y)
            # only create node if this position is free in the inflated mask
            if wall_mask.get_at((px, py)) == 0:
                node_id[(px, py)] = id_counter
                id_counter += 1

    edges: List[Tuple[int, int, float]] = []
    offsets = [(grid_step, 0), (0, grid_step), (-grid_step, 0), (0, -grid_step)]
    for (x, y), uid in list(node_id.items()):
        for dx, dy in offsets:
            nx_coord = (x + dx, y + dy)
            if nx_coord in node_id:
                vid = node_id[nx_coord]

                # verify that the straight-line segment between the two nodes does not cross any wall pixels
                def line_is_free(x1, y1, x2, y2):
                    sx = int(x2) - int(x1)
                    sy = int(y2) - int(y1)
                    steps = max(abs(sx), abs(sy), 1)
                    for i in range(1, steps + 1):
                        t = i / steps
                        ix = int(round(x1 + (x2 - x1) * t))
                        iy = int(round(y1 + (y2 - y1) * t))
                        ix = max(0, min(world_w - 1, ix))
                        iy = max(0, min(world_h - 1, iy))
                        if wall_mask.get_at((ix, iy)):
                            return False
                    return True

                if line_is_free(x, y, nx_coord[0], nx_coord[1]):
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

    # flow constraints
    node_in = {nid: [] for nid in range(len(coord_to_id))}
    node_out = {nid: [] for nid in range(len(coord_to_id))}
    for i, (u, v, w) in enumerate(edges):
        node_out[u].append(i)
        node_in[v].append(i)

    for nid in range(len(coord_to_id)):
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
    adj = {nid: [] for nid in range(len(coord_to_id))}
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


def run(image_path: Path,
        world_w: int = 1920, world_h: int = 1080,
        win_w: int = 800, win_h: int = 600,
        zoom: int = 10,
        low: int = 50, high: int = 150):
    pygame.init()
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption('Canny Walls - Zoomed View (inflated mask)')
    clock = pygame.time.Clock()

    wall_mask_vis, wall_surf_vis, bg_surf, walls_bin = load_and_prepare_mask(image_path, world_w, world_h, low, high)

    # player in world coordinates
    player_w, player_h = 3, 3
    player_x, player_y = world_w // 2, world_h // 2
    speed = 3  # world pixels per frame

    # create inflated walls (for collision/planning) based on robot footprint
    inflate_radius = max(player_w, player_h) + 1  # safety margin
    kernel = np.ones((inflate_radius * 2 + 1, inflate_radius * 2 + 1), np.uint8)
    inflated_walls = cv2.dilate(walls_bin, kernel, iterations=1)  # 0/255 image
    # convert inflated_walls to an RGBA surface for mask creation
    h, w = inflated_walls.shape
    rgba_inf = np.zeros((h, w, 4), dtype=np.uint8)
    rgba_inf[:, :, 0] = inflated_walls
    rgba_inf[:, :, 1] = inflated_walls
    rgba_inf[:, :, 2] = inflated_walls
    rgba_inf[:, :, 3] = inflated_walls
    inflated_surf = pygame.image.frombuffer(rgba_inf.tobytes(), (w, h), 'RGBA')
    inflated_mask = pygame.mask.from_surface(inflated_surf)

    # robot sprite and mask
    try:
        import robot as robot_module

        def make_robot_sprite(w, h):
            surf = pygame.Surface((w, h), flags=pygame.SRCALPHA)
            grey = 160
            surf.fill((grey, grey, grey))
            if w >= 6 and h >= 6:
                darker = (100, 100, 100)
                points = [(w - 1, h // 2), (w // 2, h // 3), (w // 2, 2 * h // 3)]
                pygame.draw.polygon(surf, darker, points)
            return surf
    except Exception:
        def make_robot_sprite(w, h):
            surf = pygame.Surface((w, h), flags=pygame.SRCALPHA)
            grey = 160
            surf.fill((grey, grey, grey))
            if w >= 6 and h >= 6:
                darker = (100, 100, 100)
                points = [(w - 1, h // 2), (w // 2, h // 3), (w // 2, 2 * h // 3)]
                pygame.draw.polygon(surf, darker, points)
            return surf

    player_surf = make_robot_sprite(player_w, player_h)
    player_mask = pygame.mask.from_surface(player_surf)

    # path following state
    path_waypoints: List[Tuple[int, int]] = []
    path_index = 0

    running = True
    show_walls = True

    try:
        font = pygame.font.SysFont(None, 20)
    except Exception:
        pygame.font.init()
        font = pygame.font.SysFont(None, 20)

    minimap_w = min(240, win_w // 4)
    minimap_h = min(160, win_h // 4)
    minimap_rect = pygame.Rect(win_w - minimap_w - 8, 8, minimap_w, minimap_h)
    factor_x = minimap_w / world_w
    factor_y = minimap_h / world_h

    # graph cache uses inflated_mask (safe)
    graph_cache = None  # (coord_to_id, edges)
    recent_positions = deque(maxlen=30)
    stuck_frames = 0
    collision_block_count = 0

    user_control_active = False
    user_control_last_frame = 0
    resume_path_after_frames = 10

    def can_place_at(x: float, y: float) -> bool:
        """
        Uses inflated_mask to determine if player placed at (x,y) (top-left) overlaps inflated walls.
        Tests floor/ceil around fractional values to avoid rounding jitter.
        """
        px_floor = int(math.floor(x))
        py_floor = int(math.floor(y))
        px_ceil = int(math.ceil(x))
        py_ceil = int(math.ceil(y))

        candidates = [(px_floor, py_floor)]
        if (px_ceil, py_floor) not in candidates:
            candidates.append((px_ceil, py_floor))
        if (px_floor, py_ceil) not in candidates:
            candidates.append((px_floor, py_ceil))
        if (px_ceil, py_ceil) not in candidates:
            candidates.append((px_ceil, py_ceil))
        for cx, cy in candidates:
            cx_clamped = max(0, min(world_w - player_w, cx))
            cy_clamped = max(0, min(world_h - player_h, cy))
            if inflated_mask.overlap(player_mask, (cx_clamped, cy_clamped)) is not None:
                return False
        return True

    def attempt_stepwise_move(cur_x: float, cur_y: float, dx: float, dy: float) -> Tuple[float, float, bool]:
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return cur_x, cur_y, False

        steps = max(1, int(math.ceil(max(abs(dx), abs(dy)))))
        step_dx = dx / steps
        step_dy = dy / steps

        x, y = cur_x, cur_y
        moved = False
        for i in range(steps):
            nx = x + step_dx
            ny = y + step_dy
            nx = max(0, min(world_w - player_w, nx))
            ny = max(0, min(world_h - player_h, ny))
            if can_place_at(nx, ny):
                x, y = nx, ny
                moved = True
            else:
                # try axis-aligned slide (prefer larger component)
                if abs(step_dx) >= abs(step_dy):
                    nx2 = x + step_dx
                    ny2 = y
                    if can_place_at(nx2, ny2):
                        x = nx2; moved = True; continue
                    nx3 = x
                    ny3 = y + step_dy
                    if can_place_at(nx3, ny3):
                        y = ny3; moved = True; continue
                else:
                    nx2 = x
                    ny2 = y + step_dy
                    if can_place_at(nx2, ny2):
                        y = ny2; moved = True; continue
                    nx3 = x + step_dx
                    ny3 = y
                    if can_place_at(nx3, ny3):
                        x = nx3; moved = True; continue
                break
        return x, y, moved

    def attempt_local_replan(current_x, current_y, goal_coord=None):
        nonlocal graph_cache, path_waypoints, path_index
        if goal_coord is None:
            return False
        try:
            if graph_cache is None:
                coord_to_id, edges = build_grid_graph_from_mask(inflated_mask, world_w, world_h, grid_step=16)
                graph_cache = (coord_to_id, edges)
            else:
                coord_to_id, edges = graph_cache

            start_coord, start_id = find_nearest_node(coord_to_id, (current_x, current_y))
            goal_coord, goal_id = find_nearest_node(coord_to_id, goal_coord)
            planned_ids = []
            if pulp is not None:
                planned_ids = solve_shortest_path_ilp(coord_to_id, edges, start_id, goal_id)
            if not planned_ids and nx is not None:
                planned_ids = solve_shortest_path_graph(coord_to_id, edges, start_id, goal_id)
            if planned_ids:
                id_to_coord = {nid: coord for coord, nid in coord_to_id.items()}
                waypoints = [id_to_coord[nid] for nid in planned_ids]
                path_waypoints = waypoints
                path_index = 0
                return True
        except Exception:
            return False
        return False

    frame = 0
    while running:
        frame += 1
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
                if minimap_rect.collidepoint((mx, my)):
                    wx = int((mx - minimap_rect.x) / factor_x)
                    wy = int((my - minimap_rect.y) / factor_y)

                    # ensure graph is built from inflated mask
                    if graph_cache is None:
                        coord_to_id, edges = build_grid_graph_from_mask(inflated_mask, world_w, world_h, grid_step=16)
                        graph_cache = (coord_to_id, edges)
                    else:
                        coord_to_id, edges = graph_cache

                    # ensure the clicked goal snaps to a valid node (nearest free node)
                    goal_snap, goal_id = find_nearest_node(coord_to_id, (wx, wy))
                    if goal_snap is None:
                        # no node found -> ignore
                        continue

                    start_snap, start_id = find_nearest_node(coord_to_id, (player_x, player_y))
                    planned_ids: List[int] = []
                    if pulp is not None:
                        planned_ids = solve_shortest_path_ilp(coord_to_id, edges, start_id, goal_id)
                    if not planned_ids and nx is not None:
                        planned_ids = solve_shortest_path_graph(coord_to_id, edges, start_id, goal_id)

                    if planned_ids:
                        id_to_coord = {nid: coord for coord, nid in coord_to_id.items()}
                        waypoints = [id_to_coord[nid] for nid in planned_ids]
                        path_waypoints.clear()
                        path_waypoints.extend(waypoints)
                        path_index = 0

        keys = pygame.key.get_pressed()
        user_input_now = (keys[pygame.K_a] or keys[pygame.K_LEFT] or
                          keys[pygame.K_d] or keys[pygame.K_RIGHT] or
                          keys[pygame.K_w] or keys[pygame.K_UP] or
                          keys[pygame.K_s] or keys[pygame.K_DOWN])

        if user_input_now:
            user_control_active = True
            user_control_last_frame = frame

        if user_control_active and (frame - user_control_last_frame) > resume_path_after_frames:
            user_control_active = False

        desired_dx = 0.0
        desired_dy = 0.0

        if user_control_active:
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                desired_dx -= speed
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                desired_dx += speed
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                desired_dy -= speed
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                desired_dy += speed
        else:
            if path_waypoints and path_index < len(path_waypoints):
                tx, ty = path_waypoints[path_index]
                vx = tx - player_x
                vy = ty - player_y
                dist = math.hypot(vx, vy)
                # advance threshold: be somewhat generous to avoid hovering because of thin differences
                advance_thresh = max(1.0, 0.5 * 16)  # half grid step (8) is a safe threshold for this grid
                if dist < advance_thresh:
                    path_index += 1
                else:
                    desired_dx = (vx / dist) * speed
                    desired_dy = (vy / dist) * speed

        new_x, new_y, moved_flag = attempt_stepwise_move(player_x, player_y, desired_dx, desired_dy)
        if moved_flag:
            player_x, player_y = new_x, new_y
            collision_block_count = 0
        else:
            collision_block_count += 1
            if not user_control_active and collision_block_count > 2 and path_waypoints:
                goal_coord = path_waypoints[-1]
                if attempt_local_replan(player_x, player_y, goal_coord=goal_coord):
                    collision_block_count = 0
                else:
                    # small nudges more aggressive now (but only after a couple blocks)
                    nudges = [(speed, 0), (-speed, 0), (0, speed), (0, -speed),
                              (speed, speed), (-speed, -speed), (speed, -speed), (-speed, speed)]
                    freed = False
                    for nx_off, ny_off in nudges:
                        cand_x = max(0, min(world_w - player_w, player_x + nx_off))
                        cand_y = max(0, min(world_h - player_h, player_y + ny_off))
                        if can_place_at(cand_x, cand_y):
                            player_x = cand_x
                            player_y = cand_y
                            recent_positions.clear()
                            stuck_frames = 0
                            collision_block_count = 0
                            freed = True
                            break
                    if not freed:
                        # drop path after repeated failures
                        path_waypoints = []
                        path_index = 0
                        collision_block_count = 0

        recent_positions.append((player_x, player_y))
        if len(recent_positions) >= recent_positions.maxlen:
            x0, y0 = recent_positions[0]
            x1, y1 = recent_positions[-1]
            net_disp = math.hypot(x1 - x0, y1 - y0)
            if net_disp < 2.0:
                stuck_frames += 1
            else:
                stuck_frames = 0

        # camera calculations
        crop_w = max(1, win_w // zoom)
        crop_h = max(1, win_h // zoom)
        player_cx = player_x + player_w // 2
        player_cy = player_y + player_h // 2
        cam_x = int(player_cx - crop_w // 2)
        cam_y = int(player_cy - crop_h // 2)
        cam_x = max(0, min(world_w - crop_w, cam_x))
        cam_y = max(0, min(world_h - crop_h, cam_y))
        cam_rect = pygame.Rect(cam_x, cam_y, crop_w, crop_h)

        try:
            bg_crop = bg_surf.subsurface(cam_rect)
            if show_walls:
                wall_crop = wall_surf_vis.subsurface(cam_rect)
            else:
                wall_crop = None
        except ValueError:
            bg_crop = pygame.Surface((crop_w, crop_h))
            bg_crop.blit(bg_surf, (0, 0), cam_rect)
            if show_walls:
                wall_crop = pygame.Surface((crop_w, crop_h), flags=pygame.SRCALPHA)
                wall_crop.blit(wall_surf_vis, (0, 0), cam_rect)
            else:
                wall_crop = None

        bg_scaled = pygame.transform.scale(bg_crop, (win_w, win_h))
        if show_walls and wall_crop is not None:
            wall_scaled = pygame.transform.scale(wall_crop, (win_w, win_h))
        else:
            wall_scaled = None

        minimap_w = min(240, win_w // 4)
        minimap_h = min(160, win_h // 4)
        minimap_surf = pygame.transform.scale(bg_surf, (minimap_w, minimap_h))
        minimap_rect = pygame.Rect(win_w - minimap_w - 8, 8, minimap_w, minimap_h)
        factor_x = minimap_w / world_w
        factor_y = minimap_h / world_h

        screen.blit(bg_scaled, (0, 0))
        if wall_scaled is not None:
            screen.blit(wall_scaled, (0, 0))

        screen.blit(minimap_surf, (minimap_rect.x, minimap_rect.y))
        wall_minimap = pygame.transform.scale(wall_surf_vis, (minimap_w, minimap_h))
        screen.blit(wall_minimap, (minimap_rect.x, minimap_rect.y))

        if path_waypoints:
            mini_points = [(minimap_rect.x + int(x * factor_x), minimap_rect.y + int(y * factor_y)) for (x, y) in path_waypoints]
            if len(mini_points) >= 2:
                pygame.draw.lines(screen, (200, 200, 200), False, mini_points, 2)

        screen_player_x = int((player_x - cam_x) * zoom)
        screen_player_y = int((player_y - cam_y) * zoom)
        screen_player_w = max(1, int(player_w * zoom))
        screen_player_h = max(1, int(player_h * zoom))
        player_vis = pygame.transform.scale(player_surf, (screen_player_w, screen_player_h))
        screen.blit(player_vis, (screen_player_x, screen_player_y))

        mode_text = "Manual" if user_control_active else ("Auto (path)" if path_waypoints else "Auto (no path)")
        hud = font.render(f'Zoom: {zoom}x  Player: ({int(player_x)},{int(player_y)})  Mode: {mode_text}', True, (255, 255, 255))
        screen.blit(hud, (8, 8))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def parse_args():
    p = argparse.ArgumentParser(description='Simple Canny->Walls pygame sim with inflated mask')
    p.add_argument('--image', type=Path, default=Path('IMG_6705.jpg'), help='Image to use (default IMG_6705.jpg)')
    p.add_argument('--width', type=int, default=1920, help='World/image width in pixels')
    p.add_argument('--height', type=int, default=1080, help='World/image height in pixels')
    p.add_argument('--win-width', type=int, default=1920, help='Window (viewport) width in pixels')
    p.add_argument('--win-height', type=int, default=1080, help='Window (viewport) height in pixels')
    p.add_argument('--zoom', type=int, default=10, help='Zoom factor (integer). Visual scale-up of the cropped region')
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
