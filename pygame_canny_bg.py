#!/usr/bin/env python3
"""
Minimal Pygame simulation: load a single image, compute Canny edges and treat edges as walls.

This variant implements a zoomed viewport that follows the player.

Behavior:
 - Loads `IMG_6705.jpg` by default (or pass --image)
 - Computes Canny edges and dilates them to create thicker walls
 - Creates a pygame mask from the walls
 - A player square (controllable with WASD / arrow keys) cannot pass through wall pixels
 - Camera/viewport follows the player and displays a zoomed-in view

Controls:
 - WASD / Arrow keys: move the player
 - SPACE: toggle wall overlay
 - Q / ESC: quit

Requires: pygame, opencv-python, numpy
"""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import pygame
import math
from typing import Dict, List, Tuple

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
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    img = cv2.resize(img, (world_w, world_h))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7), 0)
    edges = cv2.Canny(gray, low, high)

    kernel = np.ones((3, 3), np.uint8)
    walls = cv2.dilate(edges, kernel, iterations=2)

    # create an RGBA surface where wall pixels set alpha=255
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

    return wall_mask, wall_surf, display_surf


def build_grid_graph_from_mask(wall_mask: pygame.mask.Mask,
                               world_w: int, world_h: int,
                               grid_step: int = 16) -> Tuple[Dict[Tuple[int, int], int], List[Tuple[int, int, float]]]:
    """
    Build grid nodes spaced by grid_step. Return mapping coord->id and list of directed edges (u,v,weight).
    """
    node_id: Dict[Tuple[int, int], int] = {}
    id_counter = 0
    # sample grid positions with offset to avoid edge clipping
    for y in range(grid_step // 2, world_h, grid_step):
        for x in range(grid_step // 2, world_w, grid_step):
            px = min(world_w - 1, x)
            py = min(world_h - 1, y)
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
    """
    Solve ILP shortest path: returns list of node ids in path order, or empty list on failure.
    Uses pulp if available.
    """
    if pulp is None:
        return []
    prob = pulp.LpProblem('shortest_path', pulp.LpMinimize)
    # create variable for each edge index
    x_vars = {}
    for i, (u, v, w) in enumerate(edges):
        x_vars[i] = pulp.LpVariable(f'x_{i}', cat='Binary')
    # objective
    prob += pulp.lpSum([w * x_vars[i] for i, (_, _, w) in enumerate(edges)])

    # flow conservation constraints
    # build incoming/outgoing lists
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

    # solve
    solver = pulp.PULP_CBC_CMD(msg=False)
    res = prob.solve(solver)
    if res != 1:
        return []

    selected = [i for i in x_vars if pulp.value(x_vars[i]) >= 0.5]
    # build adjacency from selected edges
    adj = {nid: [] for nid in range(len(coord_to_id))}
    for i in selected:
        u, v, w = edges[i]
        adj[u].append(v)

    # walk from start to goal
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
    """
    Fallback using networkx Dijkstra to produce ordered node id path.
    """
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
    """
    world_w, world_h : size of the source image (world coordinates)
    win_w, win_h     : size of the pygame window (viewport in pixels)
    zoom             : visual zoom factor (world -> screen scale). crop size = (win_w/zoom, win_h/zoom)
    """
    pygame.init()
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption('Canny Walls - Zoomed View')
    clock = pygame.time.Clock()

    wall_mask, wall_surf, bg_surf = load_and_prepare_mask(image_path, world_w, world_h, low, high)

    # player in world coordinates (keep same logical size as previous red square: 3x3)
    player_w, player_h = 3, 3
    player_x, player_y = world_w // 2, world_h // 2
    speed = 3

    # try to use robot.py visuals if available, otherwise draw a simple greyscale robot
    try:
        import robot as robot_module
        # robot module defines BuildBot with radius; we'll create a simple sprite matching player_w/player_h
        def make_robot_sprite(w, h):
            surf = pygame.Surface((w, h), flags=pygame.SRCALPHA)
            grey = 160
            surf.fill((grey, grey, grey))
            # draw a small darker nose
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

    # minimap parameters (keep fixed)
    minimap_w = min(240, win_w // 4)
    minimap_h = min(160, win_h // 4)
    minimap_rect = pygame.Rect(win_w - minimap_w - 8, 8, minimap_w, minimap_h)
    factor_x = minimap_w / world_w
    factor_y = minimap_h / world_h

    # graph cache for planning
    graph_cache = None  # tuple (coord_to_id, edges)

    # main loop
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
                if minimap_rect.collidepoint((mx, my)):
                    # convert minimap pixel to world coords
                    wx = int((mx - minimap_rect.x) / factor_x)
                    wy = int((my - minimap_rect.y) / factor_y)
                    # build graph if needed
                    if graph_cache is None:
                        coord_to_id, edges = build_grid_graph_from_mask(wall_mask, world_w, world_h, grid_step=16)
                        graph_cache = (coord_to_id, edges)
                    else:
                        coord_to_id, edges = graph_cache

                    # find nearest nodes for start and goal
                    start_coord, start_id = find_nearest_node(coord_to_id, (player_x, player_y))
                    goal_coord, goal_id = find_nearest_node(coord_to_id, (wx, wy))
                    planned_ids: List[int] = []
                    # try ILP first
                    if pulp is not None:
                        planned_ids = solve_shortest_path_ilp(coord_to_id, edges, start_id, goal_id)
                    if not planned_ids and nx is not None:
                        planned_ids = solve_shortest_path_graph(coord_to_id, edges, start_id, goal_id)

                    if planned_ids:
                        # convert ids back to coords and set waypoints
                        id_to_coord = {nid: coord for coord, nid in coord_to_id.items()}
                        waypoints = [id_to_coord[nid] for nid in planned_ids]
                        path_waypoints.clear()
                        path_waypoints.extend(waypoints)
                        path_index = 0

        keys = pygame.key.get_pressed()
        dx = 0
        dy = 0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            dx -= speed
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            dx += speed
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            dy -= speed
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            dy += speed

        # attempt move, check collision with wall mask (world coords)
        new_x = player_x + dx
        new_y = player_y + dy

        # clamp to world bounds
        new_x = max(0, min(world_w - player_w, new_x))
        new_y = max(0, min(world_h - player_h, new_y))

        overlap = wall_mask.overlap(player_mask, (int(new_x), int(new_y)))
        if overlap is None:
            player_x = new_x
            player_y = new_y
        else:
            overlap_x = wall_mask.overlap(player_mask, (int(new_x), int(player_y)))
            overlap_y = wall_mask.overlap(player_mask, (int(player_x), int(new_y)))
            if overlap_x is None:
                player_x = new_x
            if overlap_y is None:
                player_y = new_y

        # CAMERA: determine crop rect in world coordinates
        # size of crop in world coords:
        crop_w = max(1, win_w // zoom)
        crop_h = max(1, win_h // zoom)

        # center camera on player's center
        player_cx = player_x + player_w // 2
        player_cy = player_y + player_h // 2
        cam_x = int(player_cx - crop_w // 2)
        cam_y = int(player_cy - crop_h // 2)

        # clamp camera to world bounds
        cam_x = max(0, min(world_w - crop_w, cam_x))
        cam_y = max(0, min(world_h - crop_h, cam_y))

        # get subsurfaces (must be Rect fully inside surface)
        cam_rect = pygame.Rect(cam_x, cam_y, crop_w, crop_h)

        # Crop and scale background and walls
        try:
            bg_crop = bg_surf.subsurface(cam_rect)
            if show_walls:
                wall_crop = wall_surf.subsurface(cam_rect)
            else:
                wall_crop = None
        except ValueError:
            # fallback: if subsurface fails (shouldn't), create a copy
            bg_crop = pygame.Surface((crop_w, crop_h))
            bg_crop.blit(bg_surf, (0, 0), cam_rect)
            if show_walls:
                wall_crop = pygame.Surface((crop_w, crop_h), flags=pygame.SRCALPHA)
                wall_crop.blit(wall_surf, (0, 0), cam_rect)
            else:
                wall_crop = None

        # scale crops to window size
        bg_scaled = pygame.transform.scale(bg_crop, (win_w, win_h))
        if show_walls and wall_crop is not None:
            wall_scaled = pygame.transform.scale(wall_crop, (win_w, win_h))
        else:
            wall_scaled = None

        # handle minimap click/planning (minimap top-right)
        # minimap rendering parameters
        minimap_w = min(240, win_w // 4)
        minimap_h = min(160, win_h // 4)
        minimap_surf = pygame.transform.scale(bg_surf, (minimap_w, minimap_h))
        minimap_rect = pygame.Rect(win_w - minimap_w - 8, 8, minimap_w, minimap_h)

        # draw scaled background
        screen.blit(bg_scaled, (0, 0))
        if wall_scaled is not None:
            screen.blit(wall_scaled, (0, 0))

        # draw minimap background and walls (small)
        screen.blit(minimap_surf, (minimap_rect.x, minimap_rect.y))
        # draw wall overlay on minimap
        wall_minimap = pygame.transform.scale(wall_surf, (minimap_w, minimap_h))
        screen.blit(wall_minimap, (minimap_rect.x, minimap_rect.y))

        # draw planned path on minimap
        if path_waypoints:
            # map world coords to minimap coords
            factor_x = minimap_w / world_w
            factor_y = minimap_h / world_h
            mini_points = [(minimap_rect.x + int(x * factor_x), minimap_rect.y + int(y * factor_y)) for (x, y) in path_waypoints]
            if len(mini_points) >= 2:
                pygame.draw.lines(screen, (200, 200, 200), False, mini_points, 2)

        # if following path, update movement toward next waypoint
        if path_waypoints and path_index < len(path_waypoints):
            target = path_waypoints[path_index]
            tx, ty = target
            # simple proportional move toward target
            vx = tx - player_x
            vy = ty - player_y
            dist = math.hypot(vx, vy)
            if dist < 1.0:
                path_index += 1
            else:
                move_dx = (vx / dist) * speed
                move_dy = (vy / dist) * speed
                # attempt move with collision checks
                tentative_x = player_x + move_dx
                tentative_y = player_y + move_dy
                tentative_x = max(0, min(world_w - player_w, tentative_x))
                tentative_y = max(0, min(world_h - player_h, tentative_y))
                overlap = wall_mask.overlap(player_mask, (int(tentative_x), int(tentative_y)))
                if overlap is None:
                    player_x = tentative_x
                    player_y = tentative_y
                else:
                    # collision: stop following (could implement local replan)
                    path_waypoints = []
                    path_index = 0

        # draw player scaled at viewport coordinates
        screen_player_x = int((player_x - cam_x) * zoom)
        screen_player_y = int((player_y - cam_y) * zoom)
        screen_player_w = max(1, int(player_w * zoom))
        screen_player_h = max(1, int(player_h * zoom))

        player_vis = pygame.transform.scale(player_surf, (screen_player_w, screen_player_h))
        screen.blit(player_vis, (screen_player_x, screen_player_y))

        # HUD
        hud = font.render(f'Zoom: {zoom}x  Player: ({int(player_x)},{int(player_y)})', True, (255, 255, 255))
        screen.blit(hud, (8, 8))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def parse_args():
    p = argparse.ArgumentParser(description='Simple Canny->Walls pygame sim with zoomed viewport')
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