# robot_wrapper.py
import pygame
import pymunk
import math
from pymunk.vec2d import Vec2d

class Robot:
    def __init__(self, x, y, radius=20, num_sensors=9, sensor_range=200):
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
        self.shape.friction = 0.5
        self.space.add(self.body, self.shape)

        self.angle = 0.0
        self.lidar_angles = [i * (math.pi / (num_sensors - 1)) - math.pi/2 for i in range(num_sensors)]

    def set_velocity(self, linear_speed, angular_speed):
        forward = Vec2d(1, 0).rotated(self.body.angle)
        self.body.velocity = forward * linear_speed
        self.body.angular_velocity = angular_speed

    def get_lidar_distances(self, obstacle_mask: pygame.mask.Mask, world_size) -> list[float]:
        w, h = world_size
        distances = []
        pos = self.body.position
        for rel_angle in self.lidar_angles:
            angle = self.body.angle + rel_angle
            direction = Vec2d(math.cos(angle), math.sin(angle))
            max_dist = self.sensor_range

            for dist in range(1, int(max_dist) + 1):
                check_pos = pos + direction * dist
                x, y = int(check_pos.x), int(check_pos.y)
                if not (0 <= x < w and 0 <= y < h):
                    distances.append(dist - 1)
                    break
                if obstacle_mask.get_at((x, y)):
                    distances.append(dist - 1)
                    break
            else:
                distances.append(max_dist)
        return distances

    def draw(self, surface, camera_rect, zoom):
        screen_pos = (
            int((self.body.position.x - camera_rect.x) * zoom),
            int((self.body.position.y - camera_rect.y) * zoom)
        )
        scaled_radius = int(self.radius * zoom)
        pygame.draw.circle(surface, (255, 100, 100), screen_pos, scaled_radius)
        # Draw direction indicator
        end = self.body.position + Vec2d(30, 0).rotated(self.body.angle)
        end_screen = (
            int((end.x - camera_rect.x) * zoom),
            int((end.y - camera_rect.y) * zoom)
        )
        pygame.draw.line(surface, (255, 255, 0), screen_pos, end_screen, 4)