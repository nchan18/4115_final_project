#!/usr/bin/env python3
import pygame
import cv2
import numpy as np
import math
import heapq
import sys
try:
    import pulp
except ImportError:
    print("Error: 'pulp' library is required for the Convex Solver.")
    print("Please install it using: pip install pulp")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Constants & Configuration
# -----------------------------------------------------------------------------
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
FPS = 60

# Robot
ROBOT_RADIUS = 15
ROBOT_COLOR = (0, 255, 0)
ROBOT_SPEED = 3.0
ROTATION_SPEED = 0.08
LIDAR_RANGE = 120
LIDAR_FOV = 140  # degrees
NUM_RAYS = 30    # Increased for better optimization resolution

# Navigation
GOAL_THRESHOLD = 20
LOOKAHEAD_DIST = 40
OBSTACLE_DETECT_DIST = 40  # Distance to trigger avoidance
REVERSE_DURATION = 60      # Frames to reverse

# Colors
COLOR_BG = (30, 30, 30)
COLOR_WALL = (255, 255, 255)
COLOR_PATH = (0, 200, 255)
COLOR_LIDAR_FREE = (0, 255, 0)
COLOR_LIDAR_HIT = (255, 0, 0)
COLOR_GOAL = (255, 215, 0)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def load_map(filepath):
    """
    Loads the image, converts to grayscale, thresholds to binary.
    Returns the original image (for display) and a binary mask (0=free, 255=wall).
    """
    img = cv2.imread(filepath)
    if img is None:
        # Create a default map if file not found
        img = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (300, 600), (255, 255, 255), -1)
        cv2.rectangle(img, (600, 200), (900, 400), (255, 255, 255), -1)
        # Add borders
        cv2.rectangle(img, (0,0), (WINDOW_WIDTH, WINDOW_HEIGHT), (255,255,255), 10)
    
    img = cv2.resize(img, (WINDOW_WIDTH, WINDOW_HEIGHT))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    return img, mask

def get_global_path_convex(mask, start, goal, grid_size=40):
    """
    Generates a global path using a Convex Solver (Linear Programming).
    Formulates the pathfinding as a Minimum Cost Network Flow problem.
    """
    rows, cols = mask.shape
    grid_h = rows // grid_size
    grid_w = cols // grid_size
    
    start_node = (int(start[1] // grid_size), int(start[0] // grid_size))
    goal_node = (int(goal[1] // grid_size), int(goal[0] // grid_size))
    
    if not (0 <= start_node[0] < grid_h and 0 <= start_node[1] < grid_w):
        return []
    if not (0 <= goal_node[0] < grid_h and 0 <= goal_node[1] < grid_w):
        return []
        
    # 1. Build Graph
    nodes = []
    node_indices = {}
    idx_counter = 0
    
    for r in range(grid_h):
        for c in range(grid_w):
            cy = r * grid_size + grid_size // 2
            cx = c * grid_size + grid_size // 2
            if mask[cy, cx] == 0:
                nodes.append((r, c))
                node_indices[(r, c)] = idx_counter
                idx_counter += 1
                
    if start_node not in node_indices or goal_node not in node_indices:
        return []
        
    start_idx = node_indices[start_node]
    goal_idx = node_indices[goal_node]
    
    # 2. LP Problem
    prob = pulp.LpProblem("GlobalPath_LP", pulp.LpMinimize)
    
    # 3. Variables & Objective
    vars_dict = {}
    directions = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
    objective_terms = []
    
    # Pre-calculate connections for constraints
    # node_in[v] list of vars entering v
    # node_out[u] list of vars leaving u
    node_in = {i: [] for i in range(len(nodes))}
    node_out = {i: [] for i in range(len(nodes))}
    
    for u_node in nodes:
        u_idx = node_indices[u_node]
        r, c = u_node
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (nr, nc) in node_indices:
                v_idx = node_indices[(nr, nc)]
                cost = math.sqrt(dr**2 + dc**2)
                
                var_name = f"x_{u_idx}_{v_idx}"
                var = pulp.LpVariable(var_name, 0, 1, pulp.LpContinuous)
                
                vars_dict[(u_idx, v_idx)] = var
                objective_terms.append(cost * var)
                
                node_out[u_idx].append(var)
                node_in[v_idx].append(var)
                
    prob += pulp.lpSum(objective_terms)
    
    # 4. Constraints
    for idx in range(len(nodes)):
        net_flow = pulp.lpSum(node_out[idx]) - pulp.lpSum(node_in[idx])
        rhs = 0
        if idx == start_idx: rhs = 1
        elif idx == goal_idx: rhs = -1
        prob += net_flow == rhs

    # 5. Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    if pulp.LpStatus[prob.status] != 'Optimal':
        return []
        
    # 6. Reconstruct
    path = []
    curr = start_idx
    path.append(nodes[curr])
    visited = {curr}
    
    while curr != goal_idx:
        found = False
        for neighbor_idx in range(len(nodes)):
            if (curr, neighbor_idx) in vars_dict:
                val = pulp.value(vars_dict[(curr, neighbor_idx)])
                if val and val > 0.9:
                    if neighbor_idx in visited: continue
                    curr = neighbor_idx
                    path.append(nodes[curr])
                    visited.add(curr)
                    found = True
                    break
        if not found: break
            
    pixel_path = []
    for r, c in path:
        pixel_path.append((c * grid_size + grid_size//2, r * grid_size + grid_size//2))
        
    return pixel_path

# -----------------------------------------------------------------------------
# Robot Class
# -----------------------------------------------------------------------------
class Robot:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.radius = ROBOT_RADIUS
        self.lidar_readings = []
        
        # Navigation State
        self.path = []
        self.path_index = 0
        self.state = "FOLLOW_PATH" # FOLLOW_PATH, AVOID_OBSTACLE, REVERSING
        self.reverse_timer = 0
        self.avoid_target = None
        
    def update_lidar(self, mask):
        """
        Cast rays to detect obstacles (walls).
        Returns a list of (distance, point, is_hit)
        """
        self.lidar_readings = []
        start_angle = self.theta - math.radians(LIDAR_FOV / 2)
        step_angle = math.radians(LIDAR_FOV) / (NUM_RAYS - 1)
        
        for i in range(NUM_RAYS):
            angle = start_angle + i * step_angle
            
            # Raycast
            for r in range(0, LIDAR_RANGE, 2):
                px = int(self.x + r * math.cos(angle))
                py = int(self.y + r * math.sin(angle))
                
                if not (0 <= px < WINDOW_WIDTH and 0 <= py < WINDOW_HEIGHT):
                    self.lidar_readings.append((r, (px, py), True))
                    break
                
                if mask[py, px] > 0:
                    self.lidar_readings.append((r, (px, py), True))
                    break
            else:
                # Max range, no hit
                px = int(self.x + LIDAR_RANGE * math.cos(angle))
                py = int(self.y + LIDAR_RANGE * math.sin(angle))
                self.lidar_readings.append((LIDAR_RANGE, (px, py), False))

    def get_target_point(self):
        """
        Finds the lookahead point on the global path.
        """
        if not self.path or self.path_index >= len(self.path):
            return None
            
        # Find point roughly LOOKAHEAD_DIST away
        for i in range(self.path_index, len(self.path)):
            px, py = self.path[i]
            dist = math.hypot(px - self.x, py - self.y)
            if dist > LOOKAHEAD_DIST:
                return (px, py)
                
        return self.path[-1]

    def check_blocked(self):
        """
        Check if LIDAR detects an obstacle directly ahead/too close.
        """
        min_dist = float('inf')
        for dist, _, is_hit in self.lidar_readings:
            if is_hit:
                min_dist = min(min_dist, dist)
        
        return min_dist < OBSTACLE_DETECT_DIST

    def calculate_tangential_optimization(self, goal):
        """
        Performs local Tangential Optimization.
        Objective: Maximize J(theta) = cos(theta - theta_goal)
        Constraint: Ray(theta) is free (distance > threshold)
        """
        goal_angle = math.atan2(goal[1] - self.y, goal[0] - self.x)
        
        best_angle = None
        max_score = -float('inf')
        
        start_angle = self.theta - math.radians(LIDAR_FOV / 2)
        step_angle = math.radians(LIDAR_FOV) / (NUM_RAYS - 1)
        
        # Iterate over all rays (Discrete Optimization)
        for i in range(NUM_RAYS):
            dist, point, is_hit = self.lidar_readings[i]
            angle = start_angle + i * step_angle
            
            # Constraint Check
            if dist < OBSTACLE_DETECT_DIST:
                continue # Infeasible direction
                
            # Objective Function
            score = math.cos(angle - goal_angle)
            
            if score > max_score:
                max_score = score
                best_angle = angle
                
        if best_angle is not None:
            tx = self.x + LOOKAHEAD_DIST * math.cos(best_angle)
            ty = self.y + LOOKAHEAD_DIST * math.sin(best_angle)
            return (tx, ty)
        return None

    def move(self, target, reverse=False):
        """
        Move towards target (x, y).
        If reverse is True, move backwards.
        """
        dx = target[0] - self.x
        dy = target[1] - self.y
        target_angle = math.atan2(dy, dx)
        
        if reverse:
            # Move opposite to orientation
            self.x -= ROBOT_SPEED * 0.5 * math.cos(self.theta)
            self.y -= ROBOT_SPEED * 0.5 * math.sin(self.theta)
            # No rotation while reversing in this simple model, or maybe slight random turn
            return

        # Angular difference
        diff = target_angle - self.theta
        # Normalize to [-pi, pi]
        diff = (diff + math.pi) % (2 * math.pi) - math.pi
        
        # Turn
        if abs(diff) > ROTATION_SPEED:
            self.theta += ROTATION_SPEED if diff > 0 else -ROTATION_SPEED
        else:
            self.theta = target_angle
            
        # Move forward
        self.x += ROBOT_SPEED * math.cos(self.theta)
        self.y += ROBOT_SPEED * math.sin(self.theta)

    def update(self, mask, goal):
        self.update_lidar(mask)
        
        dist_to_goal = math.hypot(goal[0] - self.x, goal[1] - self.y)
        if dist_to_goal < GOAL_THRESHOLD:
            print("Goal Reached!")
            return True # Done

        # State Machine
        if self.state == "REVERSING":
            self.reverse_timer -= 1
            # Just back up
            self.move((0,0), reverse=True) # Target doesn't matter for simple reverse
            if self.reverse_timer <= 0:
                self.state = "FOLLOW_PATH"
                # Optionally re-plan here if we had a dynamic planner
                
        elif self.state == "AVOID_OBSTACLE":
            # Check if clear
            if not self.check_blocked():
                self.state = "FOLLOW_PATH"
            else:
                # Tangential Optimization
                target = self.calculate_tangential_optimization(goal)
                if target:
                    self.move(target)
                else:
                    # Optimization failed (no feasible solution) -> Reverse
                    self.state = "REVERSING"
                    self.reverse_timer = REVERSE_DURATION
                
                # Safety check
                min_dist = min([r[0] for r in self.lidar_readings if r[2]] or [LIDAR_RANGE])
                if min_dist < 15:
                    self.state = "REVERSING"
                    self.reverse_timer = REVERSE_DURATION

        elif self.state == "FOLLOW_PATH":
            if self.check_blocked():
                print("Obstacle detected! Switching to Tangential Optimization.")
                self.state = "AVOID_OBSTACLE"
            else:
                # Normal path following
                target = self.get_target_point()
                if target:
                    self.move(target)
                    
                    # Update path index (simple progress check)
                    # If we are close to the current target index, advance
                    if self.path_index < len(self.path):
                        px, py = self.path[self.path_index]
                        if math.hypot(px - self.x, py - self.y) < LOOKAHEAD_DIST:
                            self.path_index += 1
                else:
                    # No path or end of path
                    pass
                    
        return False

    def draw(self, surface):
        # Draw Robot Body
        pygame.draw.circle(surface, ROBOT_COLOR, (int(self.x), int(self.y)), self.radius)
        # Draw Heading
        end_x = self.x + self.radius * math.cos(self.theta)
        end_y = self.y + self.radius * math.sin(self.theta)
        pygame.draw.line(surface, (0,0,0), (int(self.x), int(self.y)), (int(end_x), int(end_y)), 2)
        
        # Draw LIDAR
        for dist, point, is_hit in self.lidar_readings:
            color = COLOR_LIDAR_HIT if is_hit else COLOR_LIDAR_FREE
            pygame.draw.line(surface, color, (int(self.x), int(self.y)), point, 1)

# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Robot Navigation - Tangential Avoidance")
    clock = pygame.time.Clock()

    # Load Map
    # Replace with your actual image path
    map_path = "videos/frame_0000.jpg" 
    # If file doesn't exist, the loader creates a dummy map
    bg_img, mask = load_map(map_path)
    
    # Convert bg_img to pygame surface
    bg_surf = pygame.image.frombuffer(bg_img.tobytes(), bg_img.shape[1::-1], "BGR")

    # Setup
    start_pos = (100, 100)
    goal_pos = (1000, 600)
    
    # Initial Global Plan
    print("Generating Global Path with Convex Solver (LP)...")
    path = get_global_path_convex(mask, start_pos, goal_pos)
    if not path:
        print("Warning: No initial path found!")
    else:
        print(f"Path found with {len(path)} nodes.")

    robot = Robot(start_pos[0], start_pos[1], 0)
    robot.path = path

    running = True
    while running:
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Set new goal on click
                mx, my = pygame.mouse.get_pos()
                goal_pos = (mx, my)
                print(f"New Goal: {goal_pos}")
                path = get_global_path_convex(mask, (robot.x, robot.y), goal_pos)
                robot.path = path
                robot.path_index = 0
                robot.state = "FOLLOW_PATH"

        # Update
        robot.update(mask, goal_pos)

        # Draw
        screen.blit(bg_surf, (0, 0))
        
        # Draw Path
        if robot.path and len(robot.path) > 1:
            pygame.draw.lines(screen, COLOR_PATH, False, robot.path, 3)
            
        # Draw Goal
        pygame.draw.circle(screen, COLOR_GOAL, goal_pos, 10)
        
        # Draw Robot
        robot.draw(screen)
        
        # Draw Info
        font = pygame.font.SysFont("Arial", 18)
        state_text = font.render(f"State: {robot.state}", True, (255, 255, 255))
        screen.blit(state_text, (10, 10))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
