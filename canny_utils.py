"""Utilities for loading image, computing Canny walls and minimap helpers.
"""
from pathlib import Path
import math
import cv2
import numpy as np
import pygame


def load_and_prepare_mask(image_path: Path, world_w: int, world_h: int, low: int, high: int):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    img = cv2.resize(img, (world_w, world_h))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(gray, low, high)

    kernel = np.ones((3, 3), np.uint8)
    walls = cv2.dilate(edges, kernel, iterations=2)

    h, w = walls.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = walls
    rgba[:, :, 1] = walls
    rgba[:, :, 2] = walls
    rgba[:, :, 3] = walls
    wall_surf = pygame.image.frombuffer(rgba.tobytes(), (w, h), 'RGBA')
    wall_mask = pygame.mask.from_surface(wall_surf)

    # displayable background (dimmed original + green walls highlight)
    color_edges = np.zeros_like(img)
    color_edges[:, :, 1] = walls
    dimmed = (img * 0.4).astype(np.uint8)
    display = cv2.add(dimmed, color_edges)
    display_surf = pygame.image.frombuffer(cv2.cvtColor(display, cv2.COLOR_BGR2RGB).tobytes(), (w, h), 'RGB')

    walls_bin = (walls > 0).astype(np.uint8) * 255
    return wall_mask, wall_surf, display_surf, walls_bin


def downsample_occupancy_maxpool(walls_bin: np.ndarray, minimap_w: int, minimap_h: int) -> np.ndarray:
    world_h, world_w = walls_bin.shape
    pool_x = max(1, math.ceil(world_w / minimap_w))
    pool_y = max(1, math.ceil(world_h / minimap_h))
    pool = max(pool_x, pool_y)
    kernel = np.ones((pool, pool), np.uint8)
    pooled = cv2.dilate(walls_bin, kernel, iterations=1)
    mini = cv2.resize(pooled, (minimap_w, minimap_h), interpolation=cv2.INTER_NEAREST)
    mini = (mini > 0).astype(np.uint8) * 255
    return mini


def make_minimap_display(bg_display: np.ndarray, walls_bin: np.ndarray, minimap_w: int, minimap_h: int):
    mini_bg = cv2.resize(cv2.cvtColor(bg_display, cv2.COLOR_BGR2RGB), (minimap_w, minimap_h), interpolation=cv2.INTER_AREA)
    pooled = downsample_occupancy_maxpool(walls_bin, minimap_w, minimap_h)
    overlay = mini_bg.copy()
    mask_idx = pooled > 0
    overlay[mask_idx, 0] = (overlay[mask_idx, 0] * 0.4).astype(np.uint8)
    overlay[mask_idx, 1] = 255
    surf = pygame.image.frombuffer(overlay.tobytes(), (minimap_w, minimap_h), 'RGB')
    return surf, pooled
