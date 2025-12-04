"""Helpers to place random static objects unknown to the planner."""
import random
import pygame
from typing import Tuple, List


from typing import Optional
import numpy as np


def place_random_static_objects(surface_size: Tuple[int, int], wall_mask: pygame.mask.Mask,
                                num_rect: int = 8, num_circ: int = 6,
                                seed: int = None,
                                allowed_mask: Optional['np.ndarray'] = None,
                                grid_cell: int | Tuple[int, int] = 1,
                                max_cells: int = 2) -> Tuple[pygame.Surface, List]:
    """Place small static objects constrained to allowed_mask (white/hallways) and
    limited to at most max_cells x max_cells grid cells (each cell = grid_cell pixels).

    allowed_mask should be a 2D uint8 numpy array with 1 for allowed placement pixels.
    """
    import numpy as np
    from typing import Optional

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    world_w, world_h = surface_size
    static_surf = pygame.Surface((world_w, world_h), flags=pygame.SRCALPHA)
    static_surf.fill((0, 0, 0, 0))
    objects = []

    max_tries = 200
    # allow grid_cell to be scalar (square cell) or a (cell_w, cell_h) tuple
    if isinstance(grid_cell, (tuple, list)):
        cell_w, cell_h = int(grid_cell[0]), int(grid_cell[1])
    else:
        cell_w = cell_h = int(grid_cell)
    max_size_px = max(1, max_cells) * cell_w
    # Border placement caps: rectangles along world borders may be larger (up to 4x2 cells)
    BORDER_MAX_W_CELLS = 4
    BORDER_MAX_H_CELLS = 2
    # Circles: maximum diameter in grid cells (2 -> radius = 1 cell)
    CIRCLE_MAX_DIAMETER_CELLS = 2

    # Helper to check allowed area (if provided)
    def is_allowed_center(cx: int, cy: int) -> bool:
        if allowed_mask is None:
            return True
        if cx < 0 or cy < 0 or cx >= world_w or cy >= world_h:
            return False
        return bool(allowed_mask[cy, cx])

    # Rectangles: either place small in-body (width 1..2 cells, height 1 cell) or occasionally
    # place along a world border with larger allowed size (up to BORDER_MAX_W_CELLS x BORDER_MAX_H_CELLS).
    for _ in range(num_rect):
        for attempt in range(max_tries):
            place_on_border = random.random() < 0.15  # 15% of rects try to hug a border now
            if place_on_border:
                cells_w = random.randint(1, BORDER_MAX_W_CELLS)
                cells_h = random.randint(1, BORDER_MAX_H_CELLS)
            else:
                # make in-body rects smaller: width 1..min(2,max_cells), height fixed to 1 cell
                cells_w = random.randint(1, min(2, max(1, max_cells)))
                cells_h = 1
            rw = cells_w * cell_w
            rh = cells_h * cell_h
            if rw >= world_w or rh >= world_h:
                continue
            if place_on_border:
                side = random.choice(['left', 'right', 'top', 'bottom'])
                if side == 'left':
                    rx = 0
                    ry = random.randint(0, world_h - rh - 1)
                elif side == 'right':
                    rx = world_w - rw - 1
                    ry = random.randint(0, world_h - rh - 1)
                elif side == 'top':
                    ry = 0
                    rx = random.randint(0, world_w - rw - 1)
                else:  # bottom
                    ry = world_h - rh - 1
                    rx = random.randint(0, world_w - rw - 1)
            else:
                rx = random.randint(0, world_w - rw - 1)
                ry = random.randint(0, world_h - rh - 1)
            # center must be in allowed area
            cx = rx + rw // 2
            cy = ry + rh // 2
            if not is_allowed_center(cx, cy):
                continue
            # ensure not overlapping walls
            test_pts = [(rx + rw//2, ry + rh//2), (rx, ry), (rx+rw-1, ry), (rx, ry+rh-1), (rx+rw-1, ry+rh-1)]
            bad = False
            for (tx, ty) in test_pts:
                if tx < 0 or ty < 0 or tx >= world_w or ty >= world_h or wall_mask.get_at((tx, ty)):
                    bad = True
                    break
            if not bad:
                pygame.draw.rect(static_surf, (200, 200, 200, 255), (rx, ry, rw, rh))
                objects.append(("rect", (rx, ry, rw, rh)))
                break

    # Circles: small radii up to max_cells * min(cell_w,cell_h) // 2
    for _ in range(num_circ):
        for attempt in range(max_tries):
            # enforce max diameter of CIRCLE_MAX_DIAMETER_CELLS cells
            max_diam_cells = min(max(1, max_cells), CIRCLE_MAX_DIAMETER_CELLS)
            cells_diam = random.randint(1, max_diam_cells)
            # radius in pixels: (diameter_cells * min(cell_w,cell_h)) / 2
            base_cell = min(cell_w, cell_h)
            r = max(1, int((cells_diam * base_cell) // 2))
            cx = random.randint(r, world_w - r - 1)
            cy = random.randint(r, world_h - r - 1)
            if not is_allowed_center(cx, cy):
                continue
            test_pts = [(cx, cy), (cx + r, cy), (cx - r, cy), (cx, cy + r), (cx, cy - r)]
            bad = False
            for (tx, ty) in test_pts:
                if tx < 0 or ty < 0 or tx >= world_w or ty >= world_h or wall_mask.get_at((tx, ty)):
                    bad = True
                    break
            if not bad:
                # draw static circle in a lighter gray (was white)
                pygame.draw.circle(static_surf, (200, 200, 200, 255), (cx, cy), r)
                objects.append(("circ", (cx, cy, r)))
                break

    return static_surf, objects
