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

    # player in world coordinates
    player_w, player_h = 3, 3
    player_x, player_y = 1371,637
    speed = 3
    player_surf = pygame.Surface((player_w, player_h), flags=pygame.SRCALPHA)
    player_surf.fill((200, 50, 50))
    player_mask = pygame.mask.from_surface(player_surf)

    running = True
    show_walls = True

    try:
        font = pygame.font.SysFont(None, 20)
    except Exception:
        pygame.font.init()
        font = pygame.font.SysFont(None, 20)

    # precompute scaled player surface size for rendering (we will scale each frame)
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    show_walls = not show_walls

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

        # draw scaled background
        screen.blit(bg_scaled, (0, 0))
        if wall_scaled is not None:
            screen.blit(wall_scaled, (0, 0))

        # draw player scaled at viewport coordinates
        screen_player_x = int((player_x - cam_x) * zoom)
        screen_player_y = int((player_y - cam_y) * zoom)
        screen_player_w = max(1, int(player_w * zoom))
        screen_player_h = max(1, int(player_h * zoom))

        player_vis = pygame.transform.scale(player_surf, (screen_player_w, screen_player_h))
        screen.blit(player_vis, (screen_player_x, screen_player_y))

        # HUD
        hud = font.render(f'Zoom: {zoom}x  Player: ({player_x},{player_y})', True, (255, 255, 255))
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