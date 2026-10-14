"""Run NMS-SIFT over image tiles with any concurrent.futures executor."""
from __future__ import annotations

import time
from concurrent.futures import Executor

import numpy as np

from .sift import KeypointTuple, detect_and_describe
from .tiling import merge_keypoints, split_tiles


def single_thread_libs() -> None:
    """Stop OpenCV from starting its own thread pool, so all parallelism is ours."""
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass


def sift_tile(tile: np.ndarray):
    """Worker: NMS-SIFT on one tile, plus the CPU seconds this thread spent on it."""
    t0 = time.thread_time()
    keypoints, descriptors = detect_and_describe(tile)
    return keypoints, descriptors, time.thread_time() - t0


def sift_tiled(image: np.ndarray, n_tiles: int, executor: Executor | None = None):
    """Split `image` into `n_tiles`, run NMS-SIFT on each (serially if no executor), merge.

    Returns (keypoints, descriptors, wall_seconds, cpu_seconds). executor.map yields
    results in submission order, so tile i is always shifted by offset i. The 2023 code
    appended results from inside each thread, in completion order, which was a race.
    """
    tiles = split_tiles(image, n_tiles)
    arrays, offsets = [t for t, _ in tiles], [o for _, o in tiles]
    t0 = time.perf_counter()
    results = list(executor.map(sift_tile, arrays)) if executor else [sift_tile(a) for a in arrays]
    wall = time.perf_counter() - t0
    keypoints: list[KeypointTuple] = merge_keypoints([r[0] for r in results], offsets)
    non_empty = [r[1] for r in results if len(r[1])]
    descriptors = np.concatenate(non_empty) if non_empty else np.empty((0, 128), np.float32)
    return keypoints, descriptors, wall, sum(r[2] for r in results)
