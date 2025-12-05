#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys
import math
from typing import List, Tuple
import random

import cv2
import numpy as np
import pygame

# import modularized helpers
from canny_utils import load_and_prepare_mask, make_minimap_display
from planning import (build_grid_graph_from_mask, find_nearest_node,
                      shortcut_and_simplify, convexify_points, line_free_between_points,
                      solve_shortest_path_ilp, solve_shortest_path_graph,
                      solve_shortest_path_dijkstra, rrt_local)
from sensors import get_lidar_distances, swept_collision_check, perimeter_collision_check
from static_objects import place_random_static_objects

# ------------------------ main simulation ------------------------

def run(image_path: Path,
        world_w: int = 1920, world_h: int = 1080,
        win_w: int = 800, win_h: int = 600,
        zoom: int = 10,
        low: int = 50, high: int = 150):
    pygame.init()
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption('Canny Walls Robot Simulation')
    clock = pygame.time.Clock()

    # full-res wall mask and display image + binary
    wall_mask, wall_surf, bg_surf, walls_bin = load_and_prepare_mask(image_path, world_w, world_h, low, high)

    # compute distance-to-wall map (pixels) for planning clearance checks
    try:
        free_mask = (walls_bin == 0).astype('uint8') * 255
        dist_map = cv2.distanceTransform(free_mask, cv2.DIST_L2, 5)
    except Exception:
        dist_map = None

    # ------------------ create unknown static objects
    # compute allowed mask from background: choose dark-gray room pixels (low-to-mid brightness,
    # low saturation) so static objects appear in rooms rather than bright hallways
    try:
        bg_arr_for_mask = pygame.surfarray.array3d(bg_surf)
        bg_arr_for_mask = np.transpose(bg_arr_for_mask, (1, 0, 2)).copy()
        # convert to HSV to better segment gray (low saturation) and moderate/low brightness
        #RGB limits
        RED_LOW = 70
        RED_HIGH = 100
        BLUE_LOW = 70
        BLUE_HIGH = 100
        GREEN_LOW = 70
        GREEN_HIGH = 100
        allowed_mask = (bg_arr_for_mask[:,:,0] >= RED_LOW) & (bg_arr_for_mask[:,:,0] <= RED_HIGH) & \
                        (bg_arr_for_mask[:,:,1] >= GREEN_LOW) & (bg_arr_for_mask[:,:,1] <= GREEN_HIGH) & \
                        (bg_arr_for_mask[:,:,2] >= BLUE_LOW) & (bg_arr_for_mask[:,:,2] <= BLUE_HIGH)
    except Exception:
        allowed_mask = None

    # try to place many small rects (no circles) limited to 2x2 cells
    static_surf, static_objects = place_random_static_objects((world_w, world_h), wall_mask,
                                                              num_rect=0, num_circ=0, seed=None,
                                                              allowed_mask=allowed_mask,
                                                              grid_cell=1, max_cells=16)
    static_mask = pygame.mask.from_surface(static_surf)

    # combined mask used for movement & LIDAR checks
    # create combined surface by blitting wall_surf (alpha) then static_surf (opaque)
    combined_surf = pygame.Surface((world_w, world_h), flags=pygame.SRCALPHA)
    combined_surf.fill((0, 0, 0, 0))
    # wall_surf uses wall pixels in alpha; draw as white
    try:
        # convert wall_surf to a white mask surface
        ws_arr = pygame.surfarray.array_alpha(wall_surf)
        wall_white = pygame.Surface((world_w, world_h), flags=pygame.SRCALPHA)
        wall_white.fill((255, 255, 255, 0))
        wall_px = pygame.PixelArray(wall_white)
        # paint alpha>0 pixels white opaque
        for yy in range(world_h):
            for xx in range(world_w):
                if ws_arr[xx, yy] > 0:
                    wall_px[xx, yy] = (255, 255, 255, 255)
        del wall_px
        combined_surf.blit(wall_white, (0, 0))
    except Exception:
        # fallback: blit wall_surf directly (should also work since it has alpha)
        combined_surf.blit(wall_surf, (0, 0))
    combined_surf.blit(static_surf, (0, 0))
    combined_mask = pygame.mask.from_surface(combined_surf)

    # build combined binary map (h x w) and distance transform for clearance checks
    try:
        # walls_bin is h x w; extract static alpha and transpose to h x w
        static_alpha = pygame.surfarray.array_alpha(static_surf)
        static_bin = (np.transpose(static_alpha) > 0).astype(np.uint8) * 255
        combined_bin = np.maximum(walls_bin, static_bin)
        free_mask = (combined_bin == 0).astype('uint8') * 255
        combined_dist_map = cv2.distanceTransform(free_mask, cv2.DIST_L2, 5)
    except Exception:
        combined_bin = walls_bin.copy()
        combined_dist_map = None

    # small robot sprite
    player_w = 2
    player_h = 2
    player_x = 1542
    player_y = 900
    speed = 3

    player_surf = pygame.Surface((player_w, player_h), flags=pygame.SRCALPHA)
    player_surf.fill((200, 50, 50))
    if player_w >= 3 and player_h >= 3:
        player_surf.set_at((player_w - 1, player_h // 2), (255, 200, 50))
    player_mask = pygame.mask.from_surface(player_surf)

    # lidar params (smaller beams)
    num_lidar = 5
    lidar_max = 30
    heading = 0.0

    # planning state
    path_waypoints: List[Tuple[float, float]] = []
    path_index = 0
    pending_replan_after_retreat = False
    pending_replan_consider_static = False
    pending_replan_steps = 0
    consecutive_retreat_count = 0  # Track retreat cycles to prevent infinite loops
    MAX_CONSECUTIVE_RETREATS = 3
    graph_cache: dict[Tuple[str, int], Tuple[dict, List[Tuple[int, int, float]]]] = {}
    inflated_cache: dict[Tuple[str, int], Tuple[pygame.mask.Mask, np.ndarray | None]] = {}
    current_goal: Tuple[int, int] | None = None
    frame_count = 0
    last_replan_frame = -999
    last_successful_replan_frame = -999
    # Reduced cooldown to allow more responsive replanning
    REPLAN_COOLDOWN_FRAMES = 1
    # Grace period after successful replan to follow the path without interference
    REPLAN_GRACE_FRAMES = 60  # Increased to 1 second to allow longer paths to execute

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
    factor_x = minimap_w / float(world_w)
    factor_y = minimap_h / float(world_h)

    # thresholds (tuned)
    lidar_replan_threshold = 4     # if lidar sees object closer than this -> replan (reduced from 6)
    # reduce stopping distance by 2/3 (i.e. keep only 1/3)
    lidar_stop_threshold = 1       # was 6, now 2
    swept_samples = 6              # samples for swept collision prediction
    perimeter_sample_radius = max(2, int(max(player_w, player_h) / 2))  # used for perimeter grazing checks

    while running:
        frame_count += 1

        def ensure_graph(mask_key: str, mask_obj: pygame.mask.Mask, version: int = 0, dist_map_local=None, clearance: int = 0):
            """Build or reuse a grid graph for the provided mask."""
            cache_key = (mask_key, version, clearance)
            if cache_key not in graph_cache:
                coord_to_id, edges = build_grid_graph_from_mask(mask_obj, world_w, world_h, grid_step=16,
                                                               dist_map=dist_map_local, clearance=clearance)
                graph_cache[cache_key] = (coord_to_id, edges)
            return graph_cache[cache_key]

        def segments_line_free(points: List[Tuple[float, float]], mask_obj: pygame.mask.Mask) -> bool:
            if len(points) < 2:
                return True
            for a, b in zip(points, points[1:]):
                if not line_free_between_points(a, b, mask_obj, world_w, world_h):
                    return False
            return True
        def try_local_avoidance(desired_heading: float, clearance: int = None, dist_map_local: np.ndarray | None = None) -> bool:
            """Try small lateral sidesteps to bypass a nearby obstacle.
            If a candidate sidestep is feasible, prepend it to the current path and return True.
            Prefer convexified replacement for the route when possible.
            """
            nonlocal path_waypoints, path_index, last_successful_replan_frame
            # Don't interfere with fresh replans - give them time to execute
            if frame_count - last_successful_replan_frame < REPLAN_GRACE_FRAMES:
                log(f"local avoid: skipping (grace period active, {frame_count - last_successful_replan_frame} frames since replan)")
                return False
            # Reduced lateral choices to avoid extreme sidesteps in narrow corridors
            lateral_choices = [max(8, player_w * 8), max(16, player_w * 16), max(32, player_w * 32)]
            if clearance is None:
                clearance = max(1, int(lidar_replan_threshold))
            if dist_map_local is None:
                dist_map_local = combined_dist_map
            for ld in lateral_choices:
                for sign in (1, -1):
                    # perpendicular vector to desired_heading
                    px = -math.sin(desired_heading)
                    py = math.cos(desired_heading)
                    nx_wp = player_x + px * (ld * sign)
                    ny_wp = player_y + py * (ld * sign)
                    nx_wp = max(0, min(world_w - player_w, nx_wp))
                    ny_wp = max(0, min(world_h - player_h, ny_wp))
                    # quick check: collision-free and swept path OK
                    if combined_mask.overlap(player_mask, (int(nx_wp), int(ny_wp))) is None and not swept_collision_check((int(player_x), int(player_y)), (int(nx_wp), int(ny_wp)), player_mask, combined_mask, samples=4):
                        log(f"local avoid: inserting sidestep ({int(nx_wp)},{int(ny_wp)}) lateral={ld*sign}")
                        # candidate route: sidestep then resume remaining waypoints
                        remaining = path_waypoints[path_index:] if path_waypoints else []
                        candidate = [(float(nx_wp), float(ny_wp))] + remaining
                        # try to convexify the candidate route and accept convex hull if safe
                        try_hull = convexify_points(candidate)
                        if len(try_hull) >= 2 and segments_line_free(try_hull, combined_mask) and clearance_sufficient(try_hull, clearance, dist_map_local):
                            log(f"local avoid: accepting convex hull of sidestep route ({len(try_hull)} pts)")
                            path_waypoints = [(float(x), float(y)) for x, y in try_hull]
                            path_index = 0
                            return True
                        # fallback: accept the simple sidestep waypoint if it is safe enough
                        if clearance_sufficient(candidate, clearance, dist_map_local):
                            path_waypoints = candidate
                            path_index = 0
                            return True
            return False

        def clearance_sufficient(points: List[Tuple[float, float]], clearance: int,
                                 dist_map_local: np.ndarray | None) -> bool:
            if dist_map_local is None or not points:
                return True
            for a, b in zip(points, points[1:]):
                steps = max(int(math.hypot(b[0] - a[0], b[1] - a[1])) // 2, 1)
                for i in range(steps + 1):
                    t = i / steps
                    px = a[0] + (b[0] - a[0]) * t
                    py = a[1] + (b[1] - a[1]) * t
                    ix = max(0, min(world_w - 1, int(px)))
                    iy = max(0, min(world_h - 1, int(py)))
                    if dist_map_local[iy, ix] < clearance:
                        return False
            return True

        def log(msg: str) -> None:
            pass

        def attempt_replan(force: bool = False, consider_static: bool = False, _retreat: bool = False) -> bool:
            """Try to plan from current player position to `current_goal`.

            Returns True if a new plan was set.
            _retreat: internal flag to indicate this call is a retreat-attempt and
                      should not trigger further retreat recursion."""
            nonlocal path_waypoints, path_index, current_goal, last_replan_frame, last_successful_replan_frame, player_x, player_y, heading
            nonlocal pending_replan_after_retreat, pending_replan_consider_static, pending_replan_steps, consecutive_retreat_count
            if not force and not _retreat and frame_count - last_replan_frame < REPLAN_COOLDOWN_FRAMES:
                log("replan skipped due to cooldown")
                return False
            last_replan_frame = frame_count
            if current_goal is None:
                log("no current goal set; cannot replan")
                return False
            wx, wy = current_goal
            log(f"attempt_replan(force={force}, consider_static={consider_static}) goal=({wx},{wy})")

            required_clearance = max(1, int(lidar_replan_threshold))
            # If this is the initial plan (no existing waypoints), be permissive
            # and use a much smaller clearance so the planner can find an initial route
            # even through narrow corridors. Static objects are still ignored when
            # consider_static=False.
            if not path_waypoints and not consider_static:
                log("initial plan: using reduced clearance for planning")
                required_clearance = 1

            mask_configs: dict[str, dict[str, object]] = {
                'walls': {
                    'mask': wall_mask,
                    'dist_map': dist_map,
                    'bin': walls_bin,
                    'version': 0,
                },
                'combined': {
                    'mask': combined_mask,
                    'dist_map': combined_dist_map,
                    'bin': combined_bin,
                    'version': 0,
                },
            }

            def get_inflated_mask(base_key: str) -> Tuple[pygame.mask.Mask | None, np.ndarray | None]:
                cache_key = (base_key, required_clearance)
                if cache_key in inflated_cache:
                    return inflated_cache[cache_key]
                base_info = mask_configs.get(base_key)
                if base_info is None:
                    return None, None
                base_bin = base_info.get('bin')
                if base_bin is None:
                    return None, None
                try:
                    kernel = max(1, required_clearance * 2 + 1)
                    inflated = cv2.dilate(base_bin, np.ones((kernel, kernel), np.uint8), iterations=1)
                    h_inf, w_inf = inflated.shape
                    rgba = np.zeros((h_inf, w_inf, 4), dtype=np.uint8)
                    rgba[:, :, 0] = inflated
                    rgba[:, :, 1] = inflated
                    rgba[:, :, 2] = inflated
                    rgba[:, :, 3] = inflated
                    surf = pygame.image.frombuffer(rgba.tobytes(), (w_inf, h_inf), 'RGBA')
                    mask_local = pygame.mask.from_surface(surf)
                    free_mask_inf = (inflated == 0).astype('uint8') * 255
                    dist_map_inf = cv2.distanceTransform(free_mask_inf, cv2.DIST_L2, 5)
                    inflated_cache[cache_key] = (mask_local, dist_map_inf)
                    return inflated_cache[cache_key]
                except Exception:
                    return None, None

            base_key = 'combined' if consider_static else 'walls'
            candidate_configs: List[dict[str, object]] = []

            base_info = mask_configs.get(base_key)
            if base_info and base_info.get('mask') is not None:
                candidate_configs.append({
                    'key': base_key,
                    'mask': base_info['mask'],
                    'dist_map': base_info.get('dist_map'),
                    'version': base_info.get('version', 0),
                })

            inflated_mask_obj, inflated_dist_map = get_inflated_mask(base_key)
            if inflated_mask_obj is not None:
                candidate_configs.append({
                    'key': f'{base_key}_inflated',
                    'mask': inflated_mask_obj,
                    'dist_map': inflated_dist_map,
                    'version': required_clearance,
                })

            best_points: List[Tuple[int, int]] = []

            for cfg in candidate_configs:
                mask_key = cfg['key']
                mask_obj = cfg['mask']
                dist_map_local = cfg['dist_map']
                version = cfg['version']

                if mask_obj is None:
                    log(f"mask={mask_key} missing mask; skipping")
                    continue

                coord_to_id, edges = ensure_graph(mask_key, mask_obj, int(version), dist_map_local, int(version))
                start_world = (player_x + player_w / 2.0, player_y + player_h / 2.0)
                start_coord, start_id = find_nearest_node(coord_to_id, start_world)
                goal_coord, goal_id = find_nearest_node(coord_to_id, (wx, wy))
                log(f"mask={mask_key} start_world={start_world} nearest_start={start_coord} (id={start_id}) nearest_goal={goal_coord} (id={goal_id}) required_clearance={required_clearance}")

                def sample_min_dist(points: List[Tuple[float, float]]) -> Tuple[float, List[Tuple[int, int, float]]]:
                    """Return min distance along polyline and list of sampled points with their distances."""
                    if dist_map_local is None or not points:
                        return float('inf'), []
                    mins = float('inf')
                    below = []
                    for a, b in zip(points, points[1:]):
                        steps = max(int(math.hypot(b[0] - a[0], b[1] - a[1])) // 2, 1)
                        for i in range(steps + 1):
                            t = i / steps
                            px = int(max(0, min(world_w - 1, int(a[0] + (b[0] - a[0]) * t))))
                            py = int(max(0, min(world_h - 1, int(a[1] + (b[1] - a[1]) * t))))
                            d = float(dist_map_local[py, px])
                            if d < mins:
                                mins = d
                            if d < required_clearance:
                                below.append((px, py, d))
                    return mins, below
                if start_id is None or goal_id is None:
                    log(f"mask={mask_key} missing start/goal nodes (start={start_id}, goal={goal_id}); skipping")
                    continue

                log(
                    f"mask={mask_key} version={version} nodes={len(coord_to_id)} "
                    f"edges={len(edges)} start_id={start_id} goal_id={goal_id}"
                )

                planned_ids_local: List[int] = []
                solver_used = None
                try:
                    planned_ids_local = solve_shortest_path_ilp(coord_to_id, edges, start_id, goal_id)
                    if planned_ids_local:
                        solver_used = 'ILP'
                except Exception as exc:
                    log(f"mask={mask_key} ILP solver raised {exc.__class__.__name__}; falling back")
                    planned_ids_local = []

                if not planned_ids_local:
                    planned_ids_local = solve_shortest_path_graph(coord_to_id, edges, start_id, goal_id)
                    if planned_ids_local:
                        solver_used = 'networkx'

                if not planned_ids_local:
                    planned_ids_local = solve_shortest_path_dijkstra(coord_to_id, edges, start_id, goal_id)
                    if planned_ids_local:
                        solver_used = 'dijkstra'

                if not planned_ids_local:
                    log(f"mask={mask_key} no path from any solver; trying next candidate")
                    continue

                log(f"mask={mask_key} {solver_used} path length={len(planned_ids_local)}")
                id_to_coord = {nid: coord for coord, nid in coord_to_id.items()}
                waypoints_local = [id_to_coord[nid] for nid in planned_ids_local]
                mins_orig, below_orig = sample_min_dist(waypoints_local)
                log(f"mask={mask_key} original path min_dist={mins_orig} samples_below={len(below_orig)}")
                if below_orig:
                    log(f"mask={mask_key} sample below clearance (orig) example={below_orig[0]}")
                simplified_local = shortcut_and_simplify(waypoints_local, mask_obj, world_w, world_h, dist_map_local, required_clearance)
                mins_simpl, below_simpl = sample_min_dist(simplified_local)
                log(f"mask={mask_key} simplified path min_dist={mins_simpl} samples_below={len(below_simpl)}")
                if below_simpl:
                    log(f"mask={mask_key} sample below clearance (simplified) example={below_simpl[0]}")
                if len(simplified_local) != len(waypoints_local):
                    log(
                        f"mask={mask_key} simplified path {len(waypoints_local)} -> {len(simplified_local)} points"
                    )
                if len(simplified_local) != len(waypoints_local):
                    log(
                        f"mask={mask_key} simplified path {len(waypoints_local)} -> {len(simplified_local)} points"
                    )
                hull_local = convexify_points(simplified_local)

                chosen_points: List[Tuple[int, int]] = []
                if len(hull_local) >= 2:
                    if segments_line_free(hull_local, mask_obj):
                        if clearance_sufficient(hull_local, required_clearance, dist_map_local):
                            log(f"mask={mask_key} accepted convex hull with {len(hull_local)} points")
                            chosen_points = hull_local
                        else:
                            log(f"mask={mask_key} convex hull failed clearance check")
                    else:
                        log(f"mask={mask_key} convex hull blocked by obstacles")
                else:
                    log(f"mask={mask_key} convex hull insufficient points ({len(hull_local)})")

                if not chosen_points:
                    if segments_line_free(simplified_local, mask_obj):
                        if clearance_sufficient(simplified_local, required_clearance, dist_map_local):
                            log(f"mask={mask_key} accepted simplified path with {len(simplified_local)} points")
                            chosen_points = simplified_local
                        else:
                            # Pragmatic fallback: if clearance is strict (>3) and simplified path is collision-free,
                            # accept it with relaxed clearance requirement (half of required)
                            relaxed_clearance = max(1, required_clearance // 2)
                            if required_clearance > 3 and clearance_sufficient(simplified_local, relaxed_clearance, dist_map_local):
                                log(f"mask={mask_key} accepted simplified path with relaxed clearance ({relaxed_clearance}) - {len(simplified_local)} points")
                                chosen_points = simplified_local
                            else:
                                log(f"mask={mask_key} simplified path failed clearance; rejecting")
                    else:
                        log(f"mask={mask_key} simplified path blocked; rejecting")

                if chosen_points:
                    best_points = chosen_points
                    break
                else:
                    log(f"mask={mask_key} rejected; evaluating next candidate")

            if not best_points:
                log("planning failed; no candidate produced a valid path")
                # As a last-resort, try a local RRT detour in world coords (not graph-based).
                # This is useful when inflated/combined graphs are disconnected but a
                # short local detour may exist.
                if consider_static:
                    try:
                        log("attempting local RRT detour as last-resort")
                        start_world = (player_x + player_w / 2.0, player_y + player_h / 2.0)
                        # Use very relaxed clearance for RRT to allow narrow passages (just 1 pixel)
                        rrt_clearance = 1
                        rrt_path = rrt_local(start_world, (wx, wy), combined_mask, world_w, world_h,
                                             dist_map=combined_dist_map, clearance=rrt_clearance,
                                             step=24, max_iters=4000, goal_tolerance=32, sample_margin=256)
                        if rrt_path:
                            log(f"RRT produced path with {len(rrt_path)} pts; trying to accept")
                            simplified_rrt = shortcut_and_simplify(rrt_path, combined_mask, world_w, world_h, combined_dist_map, rrt_clearance)
                            hull_rrt = convexify_points(simplified_rrt)
                            # Try convex hull first
                            if len(hull_rrt) >= 2 and segments_line_free(hull_rrt, combined_mask):
                                if clearance_sufficient(hull_rrt, rrt_clearance, combined_dist_map):
                                    log(f"RRT: accepting convex hull with {len(hull_rrt)} points")
                                    best_points = hull_rrt
                                else:
                                    log("RRT: convex hull failed clearance")
                            # Fallback to simplified path
                            if not best_points and segments_line_free(simplified_rrt, combined_mask):
                                if clearance_sufficient(simplified_rrt, rrt_clearance, combined_dist_map):
                                    log(f"RRT: accepting simplified path with {len(simplified_rrt)} points")
                                    best_points = simplified_rrt
                                else:
                                    log("RRT: simplified path failed clearance")
                            # Last resort: accept any collision-free path from RRT
                            if not best_points and len(rrt_path) >= 2:
                                log(f"RRT: accepting raw path as last resort with {len(rrt_path)} points")
                                best_points = rrt_path
                        else:
                            log("RRT: failed to find any path")
                    except Exception as exc:
                        log(f"RRT detour raised {exc.__class__.__name__}: {exc}; skipping")
                # If we're trying to consider static obstacles and planning failed,
                # attempt a small retreat (back up) to increase clearance and retry once.
                # Only retreat if RRT also failed (best_points still empty).
                if consider_static and not _retreat and not best_points and consecutive_retreat_count < MAX_CONSECUTIVE_RETREATS:
                    # Try retreat with smaller steps to avoid oscillation
                    retreat_steps = [12, 24, 48]
                    lateral_factors = [0.0]  # No lateral retreat to avoid getting stuck in corners
                    attempted = False
                    for d in retreat_steps:
                        for lf in lateral_factors:
                            # compute candidate retreat point: back along heading, plus lateral offset
                            try:
                                bx = -math.cos(heading)
                                by = -math.sin(heading)
                                # perpendicular to heading (to the left)
                                px = -by
                                py = bx
                                rx = player_x + bx * d + px * (lf * d)
                                ry = player_y + by * d + py * (lf * d)
                            except Exception:
                                rx = player_x
                                ry = player_y
                            rx = max(0, min(world_w - player_w, rx))
                            ry = max(0, min(world_h - player_h, ry))
                            # check retreat is collision-free against combined mask
                            overlap = combined_mask.overlap(player_mask, (int(rx), int(ry)))
                            swept_ok = not swept_collision_check((int(player_x), int(player_y)), (int(rx), int(ry)), player_mask, combined_mask, samples=4)
                            attempted = True
                            log(f"retreat candidate d={d} lateral={lf} -> ({int(rx)},{int(ry)}) overlap={overlap is not None} swept_ok={swept_ok}")
                            if overlap is None and swept_ok:
                                log(f"retreat candidate accepted -> enqueue retreat to ({int(rx)},{int(ry)}) and retry after driving back")
                                consecutive_retreat_count += 1
                                log(f"retreat count: {consecutive_retreat_count}/{MAX_CONSECUTIVE_RETREATS}")
                                # enqueue a short driving retreat (interpolated waypoints) instead of teleporting
                                retreat_steps_n = max(2, int(d // max(1, speed)))
                                retreat_pts: List[Tuple[float, float]] = []
                                for tval in [0.33, 0.66, 1.0][:retreat_steps_n]:
                                    rx_t = player_x + (rx - player_x) * tval
                                    ry_t = player_y + (ry - player_y) * tval
                                    retreat_pts.append((float(rx_t), float(ry_t)))
                                # prepend retreat waypoints to current path (so robot will drive back)
                                path_waypoints = retreat_pts + path_waypoints[path_index:]
                                path_index = 0
                                # schedule a replan after retreat completes
                                pending_replan_after_retreat = True
                                pending_replan_consider_static = consider_static
                                pending_replan_steps = len(retreat_pts)
                                graph_cache.clear()
                                return True
                    if attempted:
                        log("retreat attempts exhausted; cannot increase clearance")
                elif consider_static and consecutive_retreat_count >= MAX_CONSECUTIVE_RETREATS:
                    log(f"max consecutive retreats ({MAX_CONSECUTIVE_RETREATS}) reached; trying lidar-guided exploration")
                    # Last resort: move toward the LIDAR direction with the farthest obstacle
                    # to explore open space and try to get back on track
                    world_center_x = player_x + player_w / 2.0
                    world_center_y = player_y + player_h / 2.0
                    lidar_scan = get_lidar_distances(world_center_x, world_center_y, heading,
                                                     wall_mask, static_mask, (world_w, world_h),
                                                     num_beams=num_lidar, max_range=lidar_max)
                    if lidar_scan:
                        # Find the beam with maximum distance (most open space)
                        max_dist = max(lidar_scan)
                        max_idx = lidar_scan.index(max_dist)
                        # Compute angle for that beam
                        half_span = math.pi / 2.0
                        rel_angle = (-half_span) + max_idx * (2 * half_span) / (num_lidar - 1)
                        explore_heading = heading + rel_angle
                        # Move a moderate distance in that direction
                        explore_dist = min(max_dist * 0.7, 80)  # 70% of free space or 80 pixels max
                        explore_x = world_center_x + math.cos(explore_heading) * explore_dist
                        explore_y = world_center_y + math.sin(explore_heading) * explore_dist
                        explore_x = max(0, min(world_w - player_w, explore_x))
                        explore_y = max(0, min(world_h - player_h, explore_y))
                        # Check if exploration waypoint is collision-free
                        if combined_mask.overlap(player_mask, (int(explore_x), int(explore_y))) is None:
                            if not swept_collision_check((int(player_x), int(player_y)), (int(explore_x), int(explore_y)),
                                                         player_mask, combined_mask, samples=6):
                                log(f"lidar-guided: moving toward open space at bearing {math.degrees(explore_heading):.1f}° (beam {max_idx}, dist={max_dist:.1f})")
                                # Create waypoints to explore toward open space, then try to resume goal
                                explore_steps = max(2, int(explore_dist // max(1, speed)))
                                explore_pts: List[Tuple[float, float]] = []
                                for tval in [0.33, 0.66, 1.0][:explore_steps]:
                                    ex_t = player_x + (explore_x - player_x) * tval
                                    ey_t = player_y + (explore_y - player_y) * tval
                                    explore_pts.append((float(ex_t), float(ey_t)))
                                path_waypoints = explore_pts
                                path_index = 0
                                # Reset retreat counter to allow future retries after exploration
                                consecutive_retreat_count = 0
                                # Schedule a replan after exploration completes
                                pending_replan_after_retreat = True
                                pending_replan_consider_static = consider_static
                                pending_replan_steps = len(explore_pts)
                                log(f"lidar-guided: scheduled replan after {len(explore_pts)} exploration waypoints")
                                return True
                        log("lidar-guided: exploration waypoint blocked; giving up on this goal")
                path_waypoints = []
                path_index = 0
                return False

            log(f"planning succeeded with {len(best_points)} waypoints")
            path_waypoints = [(float(x), float(y)) for x, y in best_points]
            path_index = 0
            last_successful_replan_frame = frame_count  # Mark successful replan for grace period
            consecutive_retreat_count = 0  # Reset retreat counter on successful plan
            return True
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
                    wx = int((mx - minimap_rect.x) / minimap_w * world_w)
                    wy = int((my - minimap_rect.y) / minimap_h * world_h)
                    current_goal = (wx, wy)
                    planned = attempt_replan(force=True, consider_static=False)
                    if not planned:
                        path_waypoints = []
                        path_index = 0
                        graph_cache.clear()

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

        # use combined_mask for manual collision checks
        overlap = combined_mask.overlap(player_mask, (int(tentative_x), int(tentative_y)))
        if overlap is None:
            # also check swept collision from current to tentative
            if swept_collision_check((int(player_x), int(player_y)), (int(tentative_x), int(tentative_y)),
                                     player_mask, combined_mask, samples=swept_samples):
                # predicted collision: do not move
                pass
            else:
                if move_dx != 0 or move_dy != 0:
                    heading = math.atan2(move_dy, move_dx) if (move_dx != 0 or move_dy != 0) else heading
                player_x = tentative_x
                player_y = tentative_y
        else:
            overlap_x = combined_mask.overlap(player_mask, (int(tentative_x), int(player_y)))
            overlap_y = combined_mask.overlap(player_mask, (int(player_x), int(tentative_y)))
            if overlap_x is None:
                # swept check for x-only
                if not swept_collision_check((int(player_x), int(player_y)), (int(tentative_x), int(player_y)),
                                             player_mask, combined_mask, samples=swept_samples):
                    player_x = tentative_x
                    if move_dx != 0:
                        heading = math.atan2(0, move_dx)
            if overlap_y is None:
                if not swept_collision_check((int(player_x), int(player_y)), (int(player_x), int(tentative_y)),
                                             player_mask, combined_mask, samples=swept_samples):
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
                # if we were performing a retreat, once retreat waypoints are consumed,
                # trigger the scheduled replan to compute a static-aware path
                if pending_replan_after_retreat and path_index >= pending_replan_steps:
                    pending_replan_after_retreat = False
                    log("retreat completed; attempting static-aware replan")
                    replanned = attempt_replan(force=True, consider_static=pending_replan_consider_static)
                    if not replanned:
                        log("post-retreat replan failed; will attempt local avoidance on next tick")
            else:
                # slow down as approaching waypoint
                step_speed = speed
                if dist < 12:
                    step_speed = max(1.0, speed * (dist / 12.0))

                desired_heading = math.atan2(vy, vx)
                step_x = (vx / dist) * step_speed
                step_y = (vy / dist) * step_speed

                world_center_x = player_x + player_w / 2.0
                world_center_y = player_y + player_h / 2.0

                # LIDAR checks against walls AND static objects
                lidar = get_lidar_distances(world_center_x, world_center_y, desired_heading,
                                            wall_mask, static_mask, (world_w, world_h),
                                            num_beams=num_lidar, max_range=lidar_max)
                # if front is very close, either stop or replan
                mid = len(lidar) // 2
                front_slice = lidar[max(0, mid-1): min(len(lidar), mid+2)]
                min_front = min(front_slice) if front_slice else lidar[mid]
                # During grace period, trust the fresh plan and be less sensitive to LIDAR warnings
                in_grace_period = (frame_count - last_successful_replan_frame < REPLAN_GRACE_FRAMES)
                effective_stop_threshold = lidar_stop_threshold if not in_grace_period else 0  # No LIDAR stops during grace
                effective_replan_threshold = lidar_replan_threshold if not in_grace_period else 2  # Very close only during grace
                
                if min_front <= effective_stop_threshold:
                    # immediate stop: do not advance, but KEEP the current goal so replanning can continue
                    path_waypoints = []
                    path_index = 0
                    graph_cache.clear()
                elif min_front <= effective_replan_threshold:
                    # object detected ahead: try automatic replan to same goal
                    # During grace period, skip replanning to trust the fresh plan
                    if not in_grace_period:
                        replanned = attempt_replan(consider_static=True)
                        if not replanned:
                            # if replan failed, try a local sidestep around the obstacle first
                            if try_local_avoidance(heading):
                                graph_cache.clear()
                            else:
                                path_waypoints = []
                                path_index = 0
                                graph_cache.clear()
                    # else: in grace period, ignore LIDAR warning and continue following path
                else:
                    tentative_x = player_x + step_x
                    tentative_y = player_y + step_y
                    tentative_x = max(0, min(world_w - player_w, tentative_x))
                    tentative_y = max(0, min(world_h - player_h, tentative_y))

                    # predictive swept collision
                    predicted = swept_collision_check((int(player_x), int(player_y)), (int(tentative_x), int(tentative_y)),
                                                      player_mask, combined_mask, samples=swept_samples)
                    if predicted:
                        # predicted collision with wall or static object: try replanning to the goal
                        replanned = attempt_replan(consider_static=True)
                        if not replanned:
                            if try_local_avoidance(desired_heading):
                                graph_cache.clear()
                            else:
                                path_waypoints = []
                                path_index = 0
                                graph_cache.clear()
                    else:
                        overlap = combined_mask.overlap(player_mask, (int(tentative_x), int(tentative_y)))
                        if overlap is None:
                            # extra perimeter check
                            if not perimeter_collision_check(int(tentative_x + player_w/2.0), int(tentative_y + player_h/2.0),
                                                             player_mask, combined_mask,
                                                             num_samples=8, radius=perimeter_sample_radius):
                                player_x = tentative_x
                                player_y = tentative_y
                                heading = desired_heading
                            else:
                                # perimeter grazing collision -> try replanning
                                replanned = attempt_replan(consider_static=True)
                                if not replanned:
                                    if try_local_avoidance(desired_heading):
                                        graph_cache.clear()
                                    else:
                                        path_waypoints = []
                                        path_index = 0
                                        graph_cache.clear()
                        else:
                            # collision -> try axis moves
                            overlap_x = combined_mask.overlap(player_mask, (int(tentative_x), int(player_y)))
                            overlap_y = combined_mask.overlap(player_mask, (int(player_x), int(tentative_y)))
                            moved = False
                            if overlap_x is None and not swept_collision_check((int(player_x), int(player_y)), (int(tentative_x), int(player_y)),
                                                                               player_mask, combined_mask, samples=swept_samples):
                                player_x = tentative_x
                                moved = True
                            if overlap_y is None and not swept_collision_check((int(player_x), int(player_y)), (int(player_x), int(tentative_y)),
                                                                               player_mask, combined_mask, samples=swept_samples):
                                player_y = tentative_y
                                moved = True
                            if moved:
                                heading = desired_heading
                            else:
                                # blocked -> try replanning to the same goal
                                replanned = attempt_replan(consider_static=True)
                                if not replanned:
                                    if try_local_avoidance(desired_heading):
                                        graph_cache.clear()
                                    else:
                                        path_waypoints = []
                                        path_index = 0
                                        graph_cache.clear()

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
            static_crop = static_surf.subsurface(cam_rect)
        except ValueError:
            bg_crop = pygame.Surface((crop_w, crop_h))
            bg_crop.blit(bg_surf, (0, 0), cam_rect)
            if show_walls:
                wall_crop = pygame.Surface((crop_w, crop_h), flags=pygame.SRCALPHA)
                wall_crop.blit(wall_surf, (0, 0), cam_rect)
            else:
                wall_crop = None
            static_crop = pygame.Surface((crop_w, crop_h), flags=pygame.SRCALPHA)
            static_crop.blit(static_surf, (0, 0), cam_rect)

        bg_scaled = pygame.transform.scale(bg_crop, (win_w, win_h))
        wall_scaled = pygame.transform.scale(wall_crop, (win_w, win_h)) if (wall_crop is not None) else None
        static_scaled = pygame.transform.scale(static_crop, (win_w, win_h))

        # build minimap display from full-res data (preserve walls via maxpool)
        bg_arr = pygame.surfarray.array3d(bg_surf)
        bg_arr = np.transpose(bg_arr, (1, 0, 2)).copy()
        minimap_surf, minimap_pooled = make_minimap_display(bg_arr, walls_bin, minimap_w, minimap_h)

        # draw frame
        screen.fill((0, 0, 0))
        screen.blit(bg_scaled, (0, 0))
        if wall_scaled is not None:
            screen.blit(wall_scaled, (0, 0))
        # draw static objects (unknown to planner visually)
        screen.blit(static_scaled, (0, 0))

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

        # draw lidar beams (now use combined static_mask)
        world_center_x = player_x + player_w / 2.0
        world_center_y = player_y + player_h / 2.0
        lidar_ranges = get_lidar_distances(world_center_x, world_center_y, heading,
                                           wall_mask, static_mask, (world_w, world_h),
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
        # draw path on minimap
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

        # HUD with basic static object count and lidar midpoint
        mid_idx = len(lidar_ranges) // 2
        mid_range = lidar_ranges[mid_idx] if lidar_ranges else 0
        hud = font.render(f'Zoom: {zoom}x  Pos: ({int(player_x)},{int(player_y)})  LIDAR(mid)={mid_range}  '
                          f'StaticObjs={len(static_objects)}', True, (255, 255, 255))
        screen.blit(hud, (8, 8))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

# ------------------------ CLI ------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Canny Walls Robot Simulation')
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
