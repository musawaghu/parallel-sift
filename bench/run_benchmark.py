"""Time NMS-SIFT on the 50 flower images under serial, threaded and multi-process execution.

Every config runs on every image, and configs are interleaved per image so slow
drift (thermals, background load) affects them all equally. Rows are appended to
the CSV as they finish, so an interrupted run resumes where it stopped.

Configs
    serial_whole      whole image, one call                    (2023 baseline)
    serial_tiles{N}   N tiles, one after another               (control: tiling without parallelism)
    threads{N}        N tiles, one thread per tile             (2023 method)
    procs{N}          N tiles, process pool with N workers     (the fix)

Columns
    wall_s        wall-clock seconds for the whole image
    work_s        summed per-tile CPU seconds (thread_time, measured inside each worker)
    cores_used    work_s / wall_s: how many cores were actually busy on average
    keypoints     keypoints after merging tiles
    kp_hash       hash of the sorted merged keypoints; must match across configs with the same N

Usage
    python bench/run_benchmark.py                                   # all configs, images 1-50
    python bench/run_benchmark.py --images 1-3 --configs threads4,procs4
    .venv-ft/Scripts/python bench/run_benchmark.py --configs serial_whole,serial_tiles4,threads4,threads16
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nms_sift.cv_compat import BACKEND  # noqa: E402
from nms_sift.parallel import sift_tile, sift_tiled, single_thread_libs  # noqa: E402

ALL_CONFIGS = ["serial_whole", "serial_tiles4", "serial_tiles16", "threads4", "threads16", "procs4", "procs16"]
FIELDS = ["runtime", "image", "config", "n_tiles", "wall_s", "work_s", "cores_used", "keypoints", "kp_hash"]


def _warmup(_: int) -> int:
    return os.getpid()


def keep_awake() -> None:
    """Ask Windows not to sleep while this process runs (released automatically on exit).

    A sleep in the middle of a timed run inflates wall time without adding CPU time.
    """
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def looks_stalled(wall: float, work: float) -> bool:
    """Every config keeps at least ~1 core busy, so wall >> CPU work means the machine paused (sleep, swap)."""
    return wall > 1.5 * work + 5


def runtime_tag() -> str:
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    return f"{platform.python_version()}{'' if gil else 't-nogil'}"


def parse_images(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def kp_hash(keypoints) -> str:
    rounded = sorted((round(x, 3), round(y, 3), round(s, 3), round(a, 2)) for x, y, s, a, *_ in keypoints)
    return hashlib.sha1(repr(rounded).encode()).hexdigest()[:12]


def run_config(config: str, image: np.ndarray, pools: dict[str, Executor]):
    if config == "serial_whole":
        t0 = time.perf_counter()
        keypoints, _, work = sift_tile(image)
        return time.perf_counter() - t0, work, keypoints, 1
    n = int(config.removeprefix("serial_tiles").removeprefix("threads").removeprefix("procs"))
    keypoints, _, wall, work = sift_tiled(image, n, pools.get(config))
    return wall, work, keypoints, n


def write_env(path: Path, startup: dict[str, float]) -> None:
    import numpy
    env = {
        "runtime": runtime_tag(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "numpy": numpy.__version__,
        "cv_backend": BACKEND,
        "pool_startup_s": startup,
    }
    try:
        import cv2
        env["opencv"] = cv2.__version__
    except ImportError:
        pass
    envs = json.loads(path.read_text()) if path.exists() else {}
    envs[env["runtime"]] = env
    path.write_text(json.dumps(envs, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default="1-50")
    ap.add_argument("--configs", default=",".join(ALL_CONFIGS))
    ap.add_argument("--out", default=str(ROOT / "results" / "timings.csv"))
    args = ap.parse_args()

    single_thread_libs()
    keep_awake()
    configs = args.configs.split(",")
    images = parse_images(args.images)
    runtime = runtime_tag()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out.exists():
        with out.open() as f:
            done = {(r["runtime"], int(r["image"]), r["config"]) for r in csv.DictReader(f)}

    # Pools are created and warmed up before any timing, so per-image numbers
    # exclude interpreter spawn cost. That one-off cost is recorded in env.json.
    pools: dict[str, Executor] = {}
    startup: dict[str, float] = {}
    for c in configs:
        if c.startswith(("threads", "procs")):
            n = int(c.removeprefix("threads").removeprefix("procs"))
            t0 = time.perf_counter()
            if c.startswith("threads"):
                pools[c] = ThreadPoolExecutor(max_workers=n)
            else:
                pools[c] = ProcessPoolExecutor(max_workers=min(n, os.cpu_count() or n), initializer=single_thread_libs)
            list(pools[c].map(_warmup, range(n)))
            startup[c] = round(time.perf_counter() - t0, 3)
    write_env(out.parent / "env.json", startup)
    print(f"runtime={runtime} backend={BACKEND} configs={configs} pool startup={startup}", flush=True)

    new_file = not out.exists()
    with out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for i in images:
            image = np.load(ROOT / "data" / "gray" / f"{i}.npy")
            hashes: dict[int, str] = {}
            for c in configs:
                if (runtime, i, c) in done:
                    continue
                wall, work, keypoints, n = run_config(c, image, pools)
                for _ in range(2):
                    if not looks_stalled(wall, work):
                        break
                    print(f"img {i:2d} {c}: wall {wall:.0f}s vs CPU {work:.0f}s looks like a stall, re-running", flush=True)
                    wall, work, keypoints, n = run_config(c, image, pools)
                h = kp_hash(keypoints)
                if n > 1 and hashes.setdefault(n, h) != h:
                    raise AssertionError(f"image {i}: {c} produced different keypoints than another {n}-tile config")
                writer.writerow({
                    "runtime": runtime, "image": i, "config": c, "n_tiles": n,
                    "wall_s": round(wall, 3), "work_s": round(work, 3), "cores_used": round(work / wall, 2),
                    "keypoints": len(keypoints), "kp_hash": h,
                })
                f.flush()
                print(f"img {i:2d} {c:15s} {wall:7.1f}s  cores={work / wall:4.2f}  kp={len(keypoints)}", flush=True)

    for p in pools.values():
        p.shutdown()


if __name__ == "__main__":
    main()
