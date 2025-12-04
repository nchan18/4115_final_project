"""LIDAR and collision helpers."""
from typing import Tuple, List
import math
import pygame


def get_lidar_distances(x: float, y: float, heading: float,
                        wall_mask: pygame.mask.Mask, static_mask: pygame.mask.Mask,
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
            if wall_mask.get_at((px, py)) or static_mask.get_at((px, py)):
                dist_hit = d - 1
                break
        distances.append(dist_hit)
    return distances


def swept_collision_check(old_xy: Tuple[float, float], new_xy: Tuple[float, float],
                          player_mask: pygame.mask.Mask, combined_mask: pygame.mask.Mask,
                          samples: int = 6) -> bool:
    x1, y1 = old_xy
    x2, y2 = new_xy
    for i in range(1, samples + 1):
        t = i / float(samples)
        ix = int(round(x1 + (x2 - x1) * t))
        iy = int(round(y1 + (y2 - y1) * t))
        if combined_mask.overlap(player_mask, (ix, iy)):
            return True
    return False


def perimeter_collision_check(cx: float, cy: float, player_mask: pygame.mask.Mask,
                              combined_mask: pygame.mask.Mask, num_samples: int = 8, radius: int = 2) -> bool:
    for k in range(num_samples):
        ang = 2 * math.pi * k / num_samples
        sx = int(round(cx + math.cos(ang) * radius))
        sy = int(round(cy + math.sin(ang) * radius))
        if combined_mask.get_at((sx, sy)):
            return True
    return False
