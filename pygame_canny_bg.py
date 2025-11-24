#!/usr/bin/env python3
"""
Canny Walls + Robot + Real Convex MPC + Pure Pursuit
APPLIED FIXES — December 2025

Fixes applied:
 - Robot stops at the actual clicked goal (final_goal)
 - Walls are added to pymunk as static collision segments (contours)
 - Robot radius reduced and wall inflation reduced
 - MPC objective includes angular error so it outputs ω
 - Contour-based static collision geometry (fast)
 - Conservative LIDAR linearization retained, safer fallbacks
"""
import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
import pygame
import math
from typing import List, Tuple, Dict

# External dependencies
try:
    import osqp
    from scipy import sparse
except ImportError:
    print("Please: pip install osqp scipy pymunk pygame opencv-python networkx")
    sys.exit(1)

try:
    import networkx as nx
except ImportError:
    nx = None

# ==================== ROBOT CLASS ====================
class Robot:
    def __init__(self, x, y, radius=8.0, num_sensors=13, sensor_range=300):
        import pymunk
        from pymunk.vec2d import Vec2d

        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.radius = radius
        self.sensor_range = sensor_range
        self.num_sensors = num_sensors

        mass = 1.0
        moment = pymunk.moment_for_circle(mass, 0, radius)
        self.body = pymunk.Body(mass, moment)
        self.body.position = x, y
        self.shape = pymunk.Circle(self.body, radius)
        self.shape.friction = 0.8
        self.space.add(self.body, self.shape)

        # LIDAR angles (fan)
        self.lidar_angles = [
            i * (math.pi / (num_sensors - 1)) - math.pi / 2
            for i in range(num_sensors)
        ]

    def set_velocity(self, linear: float, angular: float):
        # set pymunk velocities (kinematic style)
        from pymunk.vec2d import Vec2d
        forward = Vec2d(1, 0).rotated(self.body.angle)
        self.body.velocity = forward * linear
        self.body.angular_velocity = angular

    def get_lidar_distances(self, mask: pygame.mask.Mask, world_size) -> List[float]:
        w, h = world_size
        pos = self.body.position
        distances = []

        for rel_angle in self.lidar_angles:
            angle = self.body.angle + rel_angle
            dx, dy = math.cos(angle), math.sin(angle)

            for dist in range(1, int(self.sensor_range) + 1):
                px = int(pos.x + dx * dist)
                py = int(pos.y + dy * dist)
                if not (0 <= px < w and 0 <= py < h):
                    distances.append(dist - 1)
                    break
                if mask.get_at((px, py)):
                    distances.append(dist - 1)
                    break
            else:
                distances.append(self.sensor_range)
        return distances

    def draw(self, surface: pygame.Surface, camera_rect: pygame.Rect, zoom: float):
        sx = int((self.body.position.x - camera_rect.x) * zoom)
        sy = int((self.body.position.y - camera_rect.y) * zoom)
        r = max(4, int(self.radius * zoom))

        pygame.draw.circle(surface, (200, 50, 50), (sx, sy), r)
        pygame.draw.circle(surface, (255, 120, 120), (sx, sy), r, 3)

        # Direction arrow
        end_x = self.body.position.x + math.cos(self.body.angle) * (self.radius + 20)
        end_y = self.body.position.y + math.sin(self.body.angle) * (self.radius + 20)
        ex = int((end_x - camera_rect.x) * zoom)
        ey = int((end_y - camera_rect.y) * zoom)
        pygame.draw.line(surface, (255, 255, 0), (sx, sy), (ex, ey), 4)

# ==================== UTIL: create pymunk static obstacles from binary image ====================
def add_static_obstacles_from_binary(space, binary_img: np.ndarray, simplify_tol: float = 2.0, segment_radius: float = 1.0):
    """
    Convert binary obstacle mask to contour segments and add them to pymunk space as static segments.
    binary_img: 2D uint8, 255 = obstacle, 0 = free
    simplify_tol: approximation tolerance for cv2.approxPolyDP (bigger => fewer vertices)
    segment_radius: thickness of pymunk segments
    """
    import pymunk

    # find contours on binary image
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    static_body = space.static_body

    for cnt in contours:
        if len(cnt) < 3:
            continue
        # simplify contour to reduce number of segments
        epsilon = max(1.0, simplify_tol)
        approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
        pts = [(int(p[0][0]), int(p[0][1])) for p in approx]
        if len(pts) < 2:
            continue
        # add segments between consecutive approximated vertices
        for i in range(len(pts)):
            p1 = pts[i]
            p2 = pts[(i + 1) % len(pts)]
            # skip zero-length segments
            if p1 == p2:
                continue
            seg = pymunk.Segment(static_body, p1, p2, segment_radius)
            seg.friction = 1.0
            seg.elasticity = 0.0
            space.add(seg)
    # done

# ==================== FIXED CONVEX PLANNER (MPC-like single-step) ====================
def convex_lidar_planner(robot: Robot, goal: Tuple[float, float], lidar_dists: List[float], dt=1/60.0):
    """
    Small-step convex QP planner returning (v, w).
    Conservative: linearized lidar constraints (assuming current robot orientation).
    MPC objective includes angular error so optimizer will produce ω.
    """
    v_max, w_max = 220.0, 6.0
    pos = np.array(robot.body.position, dtype=float)
    theta = float(robot.body.angle)

    goal_vec = np.array(goal, dtype=float) - pos
    dist_to_goal = np.linalg.norm(goal_vec)
    if dist_to_goal < 30:
        return 0.0, 0.0
    goal_dir = goal_vec / max(dist_to_goal, 1e-6)
    goal_angle = math.atan2(goal_vec[1], goal_vec[0])
    angle_error = goal_angle - theta
    angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi

    # Build QP: minimize 0.5 * [v w]^T P [v w] + q^T [v w]
    # We want to *maximize* progress toward goal (negative linear term on v) and minimize angle error via ω.
    # P positive semidef
    P = sparse.csc_matrix(sparse.diags([1.0, 1.0]))
    # q shaped: favor forward velocity toward goal and penalize angular error (linear approx)
    # Using a linear term nudging ω toward sign(angle_error)
    q = np.array([-2.0 * goal_dir[0], -8.0 * (angle_error)])  # note: angle_error small -> linear penalty; conservative

    A_rows = []
    l_vals = []
    u_vals = []

    # LIDAR constraints: conservative linearization using current beam direction
    for i, dist in enumerate(lidar_dists):
        if dist > 100:
            continue
        beam_angle = theta + robot.lidar_angles[i]
        beam_dir_x = math.cos(beam_angle)
        beam_dir_y = math.sin(beam_angle)

        # coarse condition: only consider beams that point roughly forward (dot with forward > 0)
        forward_dot = beam_dir_x  # since forward is (cos theta, sin theta) and beam_dir uses theta, forward dot ~ cos(rel_angle)
        if forward_dot <= 0.05:
            continue

        # margin: reduce by safe buffer
        margin = max(15.0, dist - 35.0)
        # Simple linear constraint: forward_dot * v <= margin/dt
        A_rows.append([forward_dot, 0.0])
        l_vals.append(-np.inf)
        u_vals.append(margin / dt)

    # velocity and angular bounds
    A_rows += [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    l_vals += [-v_max, -v_max, -w_max, -w_max]
    u_vals += [v_max, v_max, w_max, w_max]

    # encourage at least some forward speed (soft): implement as a constraint with slack by making minimum v (soft fallback)
    min_forward = 30.0
    A_rows.append([1.0, 0.0])
    l_vals.append(min_forward)
    u_vals.append(v_max)

    if len(A_rows) <= 5:
        # free space: drive fast forward with small turning toward goal using angle_error
        return 180.0, max(-w_max, min(w_max, 6.0 * angle_error))

    A = sparse.csc_matrix(np.array(A_rows, dtype=float))
    l = np.array(l_vals, dtype=float)
    u = np.array(u_vals, dtype=float)

    prob = osqp.OSQP()
    try:
        prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, warm_start=True)
        res = prob.solve()
        if res.info.status_val == 1:
            v_opt = float(res.x[0])
            w_opt = float(res.x[1])
            # clamp for safety
            v_opt = max(-v_max, min(v_max, v_opt))
            w_opt = max(-w_max, min(w_max, w_opt))
            return v_opt, w_opt
    except Exception:
        pass

    # fallback if solver fails
    return 80.0, max(-1.5, min(1.5, 4.0 * angle_error))

# ==================== WORLD LOADING ====================
def load_and_prepare_mask(image_path: Path, world_w: int, world_h: int, low: int = 50, high: int = 150):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.resize(img, (world_w, world_h))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(gray, low, high)
    kernel = np.ones((3, 3), np.uint8)
    walls = cv2.dilate(edges, kernel, iterations=2)

    h, w = walls.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., :3] = walls[..., None]
    rgba[..., 3] = walls
    wall_surf = pygame.image.frombuffer(rgba.tobytes(), (w, h), 'RGBA')
    wall_mask = pygame.mask.from_surface(wall_surf)

    color_edges = np.zeros_like(img)
    color_edges[:, :, 1] = walls
    dimmed = (img * 0.4).astype(np.uint8)
    display = cv2.add(dimmed, color_edges)
    display_surf = pygame.image.frombuffer(cv2.cvtColor(display, cv2.COLOR_BGR2RGB).tobytes(), (w, h), 'RGB')

    walls_bin = (walls > 0).astype(np.uint8) * 255
    return wall_mask, wall_surf, display_surf, walls_bin

# ==================== GRID GRAPH (GLOBAL PLANNING) ====================
def build_grid_graph_from_mask(wall_mask, world_w, world_h, grid_step: int = 20):
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

    def line_free(x1, y1, x2, y2):
        steps = max(abs(int(x2) - int(x1)), abs(int(y2) - int(y1)), 1)
        for i in range(1, steps + 1):
            t = i / steps
            ix = int(round(x1 + (x2 - x1) * t))
            iy = int(round(y1 + (y2 - y1) * t))
            if wall_mask.get_at((max(0, min(world_w-1, ix)), max(0, min(world_h-1, iy)))):
                return False
        return True

    for (x, y), uid in node_id.items():
        for dx, dy in offsets:
            nxp, nyp = x + dx, y + dy
            if (nxp, nyp) in node_id and line_free(x, y, nxp, nyp):
                edges.append((uid, node_id[(nxp, nyp)], math.hypot(dx, dy)))

    return node_id, edges

def find_nearest_node(coord_to_id, point):
    best_d = float('inf')
    best_node = None
    best_id = None
    px, py = point
    for (x, y), nid in coord_to_id.items():
        d = (x - px) ** 2 + (y - py) ** 2
        if d < best_d:
            best_d = d
            best_node = (x, y)
            best_id = nid
    return best_node, best_id

def solve_shortest_path_graph(coord_to_id, edges, start_id, goal_id):
    if nx is None or start_id is None or goal_id is None:
        return None
    G = nx.DiGraph()
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    try:
        return nx.shortest_path(G, start_id, goal_id, weight='weight')
    except:
        return None

# ==================== MAIN ====================
def run(image_path: Path, world_w=1920, world_h=1080, win_w=1200, win_h=800, zoom=10):
    pygame.init()
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("Canny Walls Robot — Click Minimap to Set Goal")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)

    wall_mask_vis, wall_surf_vis, bg_surf, walls_bin = load_and_prepare_mask(image_path, world_w, world_h)
    # reduce inflation to reasonable value (robot radius ~8)
    kernel = np.ones((21, 21), np.uint8)
    inflated_walls = cv2.dilate(walls_bin, kernel, iterations=1)

    h, w = inflated_walls.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., :3] = inflated_walls[..., None]
    rgba[..., 3] = inflated_walls
    inflated_surf = pygame.image.frombuffer(rgba.tobytes(), (w, h), 'RGBA')
    inflated_mask = pygame.mask.from_surface(inflated_surf)

    # Create robot with smaller radius
    robot = Robot(x=world_w//2, y=world_h//2, radius=8.0, num_sensors=13, sensor_range=300)

    # add static obstacles to pymunk space from inflated_walls (contour segments)
    add_static_obstacles_from_binary(robot.space, inflated_walls, simplify_tol=3.0, segment_radius=1.0)

    path_waypoints: List[Tuple[int, int]] = []
    graph_cache = None

    minimap_w, minimap_h = 240, 160
    minimap_rect = pygame.Rect(win_w - minimap_w - 10, 10, minimap_w, minimap_h)
    factor_x = minimap_w / world_w
    factor_y = minimap_h / world_h

    manual_mode = False
    running = True

    final_goal = None  # store the real clicked goal coordinates (world pixels)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    manual_mode = not manual_mode
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if minimap_rect.collidepoint(event.pos):
                    wx = int((event.pos[0] - minimap_rect.x) / factor_x)
                    wy = int((event.pos[1] - minimap_rect.y) / factor_y)
                    final_goal = (wx, wy)  # store actual clicked goal

                    if graph_cache is None:
                        print("Building graph...")
                        coord_to_id, edges = build_grid_graph_from_mask(inflated_mask, world_w, world_h, 20)
                        graph_cache = (coord_to_id, edges)
                    else:
                        coord_to_id, edges = graph_cache

                    _, goal_id = find_nearest_node(coord_to_id, (wx, wy))
                    _, start_id = find_nearest_node(coord_to_id, robot.body.position)

                    if goal_id is not None and start_id is not None:
                        path_ids = solve_shortest_path_graph(coord_to_id, edges, start_id, goal_id)
                        if path_ids:
                            id_to_coord = {v: k for k, v in coord_to_id.items()}
                            path_waypoints = [id_to_coord[i] for i in path_ids]
                            print(f"Path found: {len(path_waypoints)} points")

        # === Control ===
        if manual_mode:
            keys = pygame.key.get_pressed()
            v = 180 if keys[pygame.K_w] else 0
            w = 5.0 if keys[pygame.K_a] else (-5.0 if keys[pygame.K_d] else 0.0)
            robot.set_velocity(v, w)
        else:
            pos = robot.body.position

            # If a final (clicked) goal exists, stop if close to it
            if final_goal is not None:
                d_final = math.hypot(final_goal[0] - pos.x, final_goal[1] - pos.y)
                if d_final < 28:
                    # reached true goal -> stop and clear waypoint path
                    robot.set_velocity(0.0, 0.0)
                    path_waypoints = []
                    final_goal = None
                    # step physics minimally to settle
                    robot.space.step(1/60.0)
                    continue

            if not path_waypoints:
                robot.set_velocity(0.0, 0.0)
            else:
                # Pure pursuit lookahead
                lookahead = 100
                target = None
                # choose first waypoint farther than lookahead along the path
                for wp in reversed(path_waypoints):
                    if math.hypot(wp[0] - pos.x, wp[1] - pos.y) > lookahead:
                        target = wp
                        break
                if target is None:
                    target = path_waypoints[-1]

                dx = target[0] - pos.x
                dy = target[1] - pos.y
                dist = math.hypot(dx, dy)

                # If the path's final point and close to it, stop (redundant with final_goal check)
                if dist < 25 and target == path_waypoints[-1]:
                    robot.set_velocity(0.0, 0.0)
                else:
                    desired_angle = math.atan2(dy, dx)
                    angle_error = desired_angle - robot.body.angle
                    angle_error = (angle_error + math.pi) % (2*math.pi) - math.pi

                    # Pure pursuit turn term (strong)
                    w_pursuit = 6.5 * angle_error

                    # get lidar (on inflated mask so planner is conservative)
                    lidar = robot.get_lidar_distances(inflated_mask, (world_w, world_h))
                    v_mpc, w_mpc = convex_lidar_planner(robot, target, lidar)

                    # Decide: if MPC requests braking (v small) or large omega request, trust it; otherwise follow pursuit
                    if v_mpc < 90 or abs(w_mpc) > 0.7:
                        robot.set_velocity(v_mpc, w_mpc)
                    else:
                        # go fast with pursuit angular rate (clamped)
                        w_cmd = max(-6.0, min(6.0, w_pursuit))
                        robot.set_velocity(180.0, w_cmd)

        # physics
        robot.space.step(1/60.0)

        # === Rendering ===
        cam_w = win_w // zoom
        cam_h = win_h // zoom
        cx = int(robot.body.position.x - cam_w // 2)
        cy = int(robot.body.position.y - cam_h // 2)
        cx = max(0, min(world_w - cam_w, cx))
        cy = max(0, min(world_h - cam_h, cy))
        cam_rect = pygame.Rect(cx, cy, cam_w, cam_h)

        try:
            bg_crop = bg_surf.subsurface(cam_rect)
        except Exception:
            bg_crop = bg_surf
        bg_scaled = pygame.transform.scale(bg_crop, (win_w, win_h))
        screen.blit(bg_scaled, (0, 0))

        if True:  # draw detected walls overlay for clarity
            try:
                wall_crop = wall_surf_vis.subsurface(cam_rect)
                wall_scaled = pygame.transform.scale(wall_crop, (win_w, win_h))
                screen.blit(wall_scaled, (0, 0))
            except:
                pass

        robot.draw(screen, cam_rect, zoom)

        # Minimap
        mini = pygame.transform.scale(bg_surf, (minimap_w, minimap_h))
        screen.blit(mini, minimap_rect)
        if path_waypoints:
            points = [(minimap_rect.x + int(x * factor_x), minimap_rect.y + int(y * factor_y)) for x, y in path_waypoints]
            pygame.draw.lines(screen, (0, 255, 255), False, points, 3)
        if final_goal is not None:
            gx = minimap_rect.x + int(final_goal[0] * factor_x)
            gy = minimap_rect.y + int(final_goal[1] * factor_y)
            pygame.draw.circle(screen, (0,255,0), (gx, gy), 6)
        pygame.draw.circle(screen, (255, 0, 0),
                          (minimap_rect.x + int(robot.body.position.x * factor_x),
                           minimap_rect.y + int(robot.body.position.y * factor_y)), 6)

        # HUD
        mode = "MANUAL (WASD)" if manual_mode else ("AUTO" if path_waypoints else "IDLE")
        text = font.render(f"{mode} | Click minimap → goal | C = toggle", True, (255, 255, 255))
        screen.blit(text, (10, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, default='IMG_6705.jpg')
    parser.add_argument('--zoom', type=int, default=10)
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Image not found: {args.image}")
        sys.exit(1)

    run(args.image, zoom=args.zoom)
