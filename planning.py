"""Planning and geometry helpers extracted from main script.
"""
from typing import Dict, List, Tuple, Optional
import heapq
import math
import pygame
import numpy as np


def build_grid_graph_from_mask(wall_mask: pygame.mask.Mask,
                               world_w: int, world_h: int,
                               grid_step: int = 16,
                               dist_map: Optional[np.ndarray] = None,
                               clearance: int = 0) -> Tuple[Dict[Tuple[int, int], int], List[Tuple[int, int, float]]]:
    node_id: Dict[Tuple[int, int], int] = {}
    id_counter = 0
    for y in range(grid_step // 2, world_h, grid_step):
        for x in range(grid_step // 2, world_w, grid_step):
            px = min(world_w - 1, x)
            py = min(world_h - 1, y)
            # Node is valid only if pixel is free and, optionally, has sufficient clearance
            pixel_free = (wall_mask.get_at((px, py)) == 0)
            has_clearance = True
            if dist_map is not None and clearance > 0:
                # clamp indices
                ix = max(0, min(world_w - 1, px))
                iy = max(0, min(world_h - 1, py))
                try:
                    has_clearance = float(dist_map[iy, ix]) >= clearance
                except Exception:
                    has_clearance = True
            if pixel_free and has_clearance:
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
                # Only add an edge if the straight-line between nodes is free of wall pixels
                # Only add an edge if the straight-line between nodes is free of wall pixels
                if line_free_between_points((x, y), (nx_coord[0], nx_coord[1]), wall_mask, world_w, world_h):
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
    # ILP solver optional; keep stub use-case - main will import pulp if present
    try:
        import pulp
    except Exception:
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
    try:
        import networkx as nx
    except Exception:
        return []
    G = nx.DiGraph()
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    try:
        node_path = nx.shortest_path(G, source=start_id, target=goal_id, weight='weight')
        return node_path
    except Exception:
        return []


def solve_shortest_path_dijkstra(coord_to_id: Dict[Tuple[int, int], int],
                                 edges: List[Tuple[int, int, float]],
                                 start_id: int, goal_id: int) -> List[int]:
    if start_id == goal_id:
        return [start_id]

    adjacency: Dict[int, List[Tuple[int, float]]] = {}
    for u, v, w in edges:
        adjacency.setdefault(u, []).append((v, w))

    dist: Dict[int, float] = {start_id: 0.0}
    prev: Dict[int, int] = {}
    heap: List[Tuple[float, int]] = [(0.0, start_id)]
    visited: set[int] = set()

    while heap:
        cur_dist, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if node == goal_id:
            break
        for neighbor, weight in adjacency.get(node, []):
            if neighbor in visited:
                continue
            nd = cur_dist + weight
            if nd < dist.get(neighbor, math.inf):
                dist[neighbor] = nd
                prev[neighbor] = node
                heapq.heappush(heap, (nd, neighbor))

    if goal_id not in prev and goal_id != start_id:
        return []

    path_ids = [goal_id]
    cur = goal_id
    while cur != start_id:
        cur = prev.get(cur)
        if cur is None:
            return []
        path_ids.append(cur)
    path_ids.reverse()
    return path_ids


def line_free_between_points(p1: Tuple[float, float], p2: Tuple[float, float],
                             wall_mask: pygame.mask.Mask, world_w: int, world_h: int) -> bool:
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


def rrt_local(start: Tuple[float, float], goal: Tuple[float, float],
              wall_mask: pygame.mask.Mask, world_w: int, world_h: int,
              dist_map: Optional[np.ndarray] = None, clearance: int = 0,
              step: int = 32, max_iters: int = 2000, goal_tolerance: int = 24,
              sample_margin: int = 160) -> List[Tuple[int, int]]:
    """Simple RRT local planner returning a list of waypoints (int coords).

    - start, goal: world coordinates (floats or ints)
    - wall_mask: pygame mask for obstacle tests
    - dist_map/clearance: used to ensure sampled edges respect clearance
    - step: extension step size in pixels
    - max_iters: max iterations
    - goal_tolerance: distance threshold to consider goal reached
    - sample_margin: how far beyond bounding box between start/goal to sample
    """
    import random

    sx, sy = float(start[0]), float(start[1])
    gx, gy = float(goal[0]), float(goal[1])

    # bounding box for sampling
    min_x = int(max(0, min(sx, gx) - sample_margin))
    max_x = int(min(world_w - 1, max(sx, gx) + sample_margin))
    min_y = int(max(0, min(sy, gy) - sample_margin))
    max_y = int(min(world_h - 1, max(sy, gy) + sample_margin))

    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    nodes: List[Tuple[float, float]] = [(sx, sy)]
    parent: List[int] = [-1]

    for it in range(max_iters):
        if random.random() < 0.1:
            rx, ry = gx, gy
        else:
            rx = random.randint(min_x, max_x)
            ry = random.randint(min_y, max_y)

        # find nearest
        nearest_idx = 0
        nd = dist(nodes[0], (rx, ry))
        for i, n in enumerate(nodes):
            dcur = dist(n, (rx, ry))
            if dcur < nd:
                nd = dcur
                nearest_idx = i

        nx, ny = nodes[nearest_idx]
        dtot = dist((nx, ny), (rx, ry))
        if dtot <= 1e-6:
            continue
        step_len = min(step, dtot)
        vx = (rx - nx) / dtot
        vy = (ry - ny) / dtot
        newx = nx + vx * step_len
        newy = ny + vy * step_len

        # collision check for segment (nearest -> new)
        if not line_free_between_points((nx, ny), (newx, newy), wall_mask, world_w, world_h):
            continue
        # clearance check along the segment
        if dist_map is not None and clearance > 0:
            ok = True
            steps = max(int(math.hypot(newx - nx, newy - ny)), 1)
            for si in range(steps + 1):
                t = si / steps
                px = int(max(0, min(world_w - 1, int(nx + (newx - nx) * t))))
                py = int(max(0, min(world_h - 1, int(ny + (newy - ny) * t))))
                try:
                    if float(dist_map[py, px]) < clearance:
                        ok = False
                        break
                except Exception:
                    ok = False
                    break
            if not ok:
                continue

        nodes.append((newx, newy))
        parent.append(nearest_idx)

        # check if we can connect to goal
        if dist((newx, newy), (gx, gy)) <= goal_tolerance:
            if line_free_between_points((newx, newy), (gx, gy), wall_mask, world_w, world_h):
                # recreate path
                path: List[Tuple[int, int]] = []
                cur = len(nodes) - 1
                while cur != -1:
                    px, py = nodes[cur]
                    path.append((int(round(px)), int(round(py))))
                    cur = parent[cur]
                path.reverse()
                path.append((int(round(gx)), int(round(gy))))
                return path

    return []


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
                          wall_mask: pygame.mask.Mask, world_w: int, world_h: int,
                          dist_map: Optional[np.ndarray] = None, clearance: int = 0) -> List[Tuple[int, int]]:
    if not path_coords:
        return []
    simplified = [path_coords[0]]
    i = 0
    n = len(path_coords)
    while i < n - 1:
        far = i + 1
        for j in range(n - 1, i, -1):
            if not line_free_between_points(path_coords[i], path_coords[j], wall_mask, world_w, world_h):
                continue
            # if a distance map and clearance are provided, ensure the straight shortcut
            # also respects clearance along its length
            if dist_map is not None and clearance > 0:
                ok = True
                steps = max(int(math.hypot(path_coords[j][0] - path_coords[i][0], path_coords[j][1] - path_coords[i][1])), 1)
                for si in range(steps + 1):
                    t = si / steps
                    px = int(max(0, min(world_w - 1, int(path_coords[i][0] + (path_coords[j][0] - path_coords[i][0]) * t))))
                    py = int(max(0, min(world_h - 1, int(path_coords[i][1] + (path_coords[j][1] - path_coords[i][1]) * t))))
                    try:
                        if float(dist_map[py, px]) < clearance:
                            ok = False
                            break
                    except Exception:
                        # if sampling fails, conservatively disallow shortcut
                        ok = False
                        break
                if not ok:
                    continue
            far = j
            break
        simplified.append(path_coords[far])
        i = far
    out = []
    for p in simplified:
        if not out or out[-1] != p:
            out.append(p)
    return out
