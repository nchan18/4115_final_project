#!/usr/bin/env python3
"""
Static S-OPT demo:
- builds a grid graph from the EPIC 2nd floor map
- solves for the shortest path from C to M
- prints path length and nominal time
- saves the waypoints so the dynamic simulator can reuse them
"""

from pathlib import Path
import math
import numpy as np
import cv2
import pygame

from canny_utils import load_and_prepare_mask
from planning import (
    build_grid_graph_from_mask,
    solve_shortest_path_dijkstra,
    find_nearest_node,
    shortcut_and_simplify,
    convexify_points,
)

IMAGE_PATH = Path("IMG_6705.jpg")      # fire-escape map
WORLD_W, WORLD_H = 1920, 1080          # match pygame_canny_bg defaults
GRID_STEP = 24                         # grid spacing in pixels
CLEARANCE_PIX = 6                      # clearance in pixels for graph nodes
LOW, HIGH = 50, 150                    # Canny thresholds (same as main script)

# NOTE: adjust these pixel coordinates to your actual closet (C) and mess (M)
C_PIXEL = (1542, 900)   # approx closet location
M_PIXEL = (960, 260)    # approx mess location in upper corridor


def build_distance_transform(walls_bin: np.ndarray) -> np.ndarray:
    """Compute distance to nearest obstacle for clearance constraints."""
    free = (walls_bin == 0).astype("uint8") * 255
    dist_map = cv2.distanceTransform(free, cv2.DIST_L2, 5)
    return dist_map


def path_length(points):
    return sum(
        math.hypot(points[i + 1][0] - points[i][0],
                   points[i + 1][1] - points[i][1])
        for i in range(len(points) - 1)
    )


def main():
    pygame.init()

    # Load walls and binary occupancy from the EPIC map
    wall_mask, wall_surf, display_surf, walls_bin = load_and_prepare_mask(
        IMAGE_PATH, WORLD_W, WORLD_H, LOW, HIGH
    )

    dist_map = build_distance_transform(walls_bin)

    # Build grid graph respecting clearance
    coord_to_id, edges = build_grid_graph_from_mask(
        wall_mask, WORLD_W, WORLD_H,
        grid_step=GRID_STEP,
        dist_map=dist_map,
        clearance=CLEARANCE_PIX,
    )

    if not coord_to_id:
        print("No free nodes were generated; check thresholds / image.")
        return

    # Map C and M to nearest graph nodes
    start_coord, start_id = find_nearest_node(coord_to_id, C_PIXEL)
    goal_coord, goal_id = find_nearest_node(coord_to_id, M_PIXEL)
    print(f"Start node at {start_coord}, goal node at {goal_coord}")

    # Solve shortest path (S-OPT)
    node_ids = solve_shortest_path_dijkstra(coord_to_id, edges, start_id, goal_id)
    if not node_ids:
        print("No path found from C to M.")
        return

    id_to_coord = {nid: xy for xy, nid in coord_to_id.items()}
    raw_path = [id_to_coord[i] for i in node_ids]

    # Optional: shortcut + convexify (matches writeup)
    simplified = shortcut_and_simplify(
        raw_path, wall_mask, WORLD_W, WORLD_H, dist_map, CLEARANCE_PIX
    )
    hull_path = convexify_points(simplified)

    L = path_length(hull_path)
    v_nom = 1.0  # pixels per simulation time unit (adjust to match your robot)
    T = L / v_nom

    print(f"S-OPT path C->M has {len(hull_path)} waypoints")
    print(f"  Length: {L:.2f} pixels  |  Nominal time: {T:.2f} time units")

    # Save C->M path so the dynamic simulator can reuse it and then reverse it for M->C
    np.savez("sopt_path_C_to_M.npz", waypoints=np.array(hull_path, dtype=np.float32))
    print("Saved waypoints to sopt_path_C_to_M.npz (reverse this for the return trip).")


if __name__ == "__main__":
    main()
