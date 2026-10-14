"""Split an image into a grid of tiles and merge per-tile keypoints back into image coordinates."""
from __future__ import annotations

import math

import numpy as np

from .sift import KeypointTuple


def split_tiles(image: np.ndarray, n_tiles: int) -> list[tuple[np.ndarray, tuple[int, int]]]:
    """Split `image` into an n x n grid (n_tiles must be a perfect square).

    Returns (tile, (x_offset, y_offset)) in row-major order. Boundaries use the same
    integer division as the 2023 notebook (img[:h//2, :w//2] etc. for 4 tiles).
    """
    side = math.isqrt(n_tiles)
    if side * side != n_tiles:
        raise ValueError(f"n_tiles must be a perfect square, got {n_tiles}")
    h, w = image.shape[:2]
    ys = [r * h // side for r in range(side + 1)]
    xs = [c * w // side for c in range(side + 1)]
    tiles = []
    for r in range(side):
        for c in range(side):
            tile = np.ascontiguousarray(image[ys[r]:ys[r + 1], xs[c]:xs[c + 1]])
            tiles.append((tile, (xs[c], ys[r])))
    return tiles


def merge_keypoints(per_tile: list[list[KeypointTuple]], offsets: list[tuple[int, int]]) -> list[KeypointTuple]:
    """Shift each tile's keypoints by its offset. `per_tile` must be in the same order as `offsets`."""
    merged = []
    for keypoints, (dx, dy) in zip(per_tile, offsets, strict=True):
        merged.extend((x + dx, y + dy, *rest) for x, y, *rest in keypoints)
    return merged
