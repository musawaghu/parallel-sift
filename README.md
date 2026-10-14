# Parallel NMS-SIFT: why more threads didn't help, and what did

Senior undergraduate research project in parallel processing for content-based image retrieval (CBIR),
rebuilt and re-run end to end.

![Speedup over serial, by method and tile count](results/speedup.png)

## TL;DR

**Hypothesis (2023):** splitting each image into 16 tiles, one thread per tile, would extract SIFT features
faster than 4 tiles.

**Result (2023):** a null result. 16 tiles were faster on 24 of 36 images, by about 1.4% on average. Neither
setup was meaningfully faster than plain serial code.

**Diagnosis:** the SIFT implementation is pure Python, and CPython's **Global Interpreter Lock** lets only one
thread run Python bytecode at a time. The threads were taking turns on one core. Measured directly, threaded
runs kept **1.0 core** busy on an 8-core CPU.

**Fix:** use processes instead of threads, so each tile gets its own interpreter and its own core.

Rerun on the same 50 images and the same CPU (AMD Ryzen 7 3700X, 8 cores / 16 threads). Speedup is the
geometric mean over 50 images, relative to whole-image serial on the same interpreter:

| Method | 4 tiles | 16 tiles | Cores busy (16 tiles) |
|---|---|---|---|
| Serial, tiles one at a time (control) | 1.05× | 1.14× | 1.0 |
| Threads, GIL (the 2023 method) | 1.04× | 1.13× | 1.0 |
| **Processes** | **3.42×** | **6.65×** | 9.4 |
| Threads, free-threaded Python 3.14t (no GIL) | 3.30× | 5.82× | 9.7 |

Two takeaways:
- **The small edge 16 tiles had in 2023 came from tiling, not threading.** Serial code on the same 16 tiles
  gets about the same speedup (1.14×), because smaller tiles lose keypoints at their borders and so do less descriptor work.
- **With real parallelism, the original hypothesis holds.** 16 tiles beat 4 because they fill all 8 cores and
  even out uneven tiles. It was the GIL, not the idea, that sank the 2023 experiment.

Full write-up with live demo: [`notebooks/parallel_sift.ipynb`](notebooks/parallel_sift.ipynb).

## Diagnosis in one chart

Each worker reports the CPU time its own thread used. Summed CPU time divided by wall time gives how many
cores were actually busy:

![Cores busy per method](results/cores_used.png)

Threads with the GIL sit at 1.0 core, the same as serial. Processes and no-GIL threads use the whole CPU.
Values above 8 come from SMT (two hardware threads per core).

![Per-image wall time](results/time_per_image.png)

## Is this the same experiment as 2023?

Yes. The original notebook was lost, but a checkpoint with the code and outputs survived
([`archive/original_2023_checkpoint.ipynb`](archive/original_2023_checkpoint.ipynb), plots stripped), along
with the slides ([`archive/2023_presentation.pdf`](archive/2023_presentation.pdf)).

- **Same images.** The 2023 code read `dataset/1.jpg … 50.jpg`. These are `image_00001 … image_00050` of
  Oxford 102 Flowers. File sizes match the slide's "Image Size" column, and NMS-SIFT is deterministic, so the
  keypoint counts reproduce:
  - **16 tiles:** 36 of 36 images match the slide exactly.
  - **4 tiles:** 44 of 50 match exactly. The other 6 are off by at most 2 keypoints out of more than 1,000,
    which is floating-point drift between OpenCV 4.7 / NumPy 1.x (2023) and OpenCV 5.0 / NumPy 2.5 (now).
- **Same algorithm.** [`nms_sift/sift.py`](nms_sift/sift.py) is the original code with its numerics unchanged.
- **Same CPU.** The threaded times now are about 0.81× the 2023 times on every image. That uniform shift is
  Python 3.14 being faster than 3.9, not a different method:

<img src="results/vs_2023.png" width="460" alt="2023 vs 2026 threaded times">

## Bugs found in the original code

1. **Result-ordering race.** Each thread called `results.append(...)` when it *finished*, but tile *i* was
   later shifted by offset *i*. Threads finish in completion order (the notebook demo shows `[3, 1, 0, 2]`), so
   merged keypoints could land in the wrong quadrant. Timings and counts were unaffected.
   [`sift_tiled`](nms_sift/parallel.py) uses `executor.map`, which keeps submission order.
2. **`cv2.KeyPoint` can't be pickled.** This broke saving results in 2023, and it also blocks
   multiprocessing. Keypoints now cross process boundaries as plain tuples.
3. **The retrieval stage crashed** (`perspectiveTransform` was given the wrong image shape). Retrieval
   accuracy is out of scope here; this study measures feature-extraction time.

## Free-threaded Python 3.14 (no GIL)

PEP 703 made the GIL optional, and Python 3.14 officially supports the free-threaded build. Findings:

- **No-GIL threads nearly match processes.** They reach 3.30× vs 3.42× at 4 tiles, and 5.82× vs 6.65× at 16
  tiles, with no pickling and no process spawning.
- **The free-threaded interpreter is about 11% slower single-threaded** (79.5 s vs 71.5 s per image). That
  accounts for most of the 16-tile gap.
- **OpenCV has no free-threaded wheels yet.** The four OpenCV calls SIFT needs are reimplemented in NumPy in
  [`nms_sift/cv_compat.py`](nms_sift/cv_compat.py). This is why the no-GIL speedups are measured against a
  serial run on the same 3.14t interpreter.

## How the benchmark works

[`bench/run_benchmark.py`](bench/run_benchmark.py) runs 7 configurations on each image, interleaved image by
image so slow drift affects them all equally:
- **Serial:** whole image, 4 tiles, 16 tiles.
- **Threads:** 4 and 16 tiles.
- **Processes:** 4 and 16 tiles.

Safeguards:
- **Fair process timing.** Process pools are created and warmed up before timing. The one-off start-up
  cost (0.3–0.5 s) is reported separately in `results/env.json`.
- **No hidden parallelism.** OpenCV's internal threading is turned off.
- **Same output.** Every parallel config must produce keypoints identical to serial on the same tiles
  (hash-checked for each image).
- **No sleep artifacts.** The run asks Windows not to sleep. Any run whose wall time far exceeds its CPU time
  (a paused machine) is redone, and the chart script rejects any that remain.

Raw data: [`results/timings.csv`](results/timings.csv) (500 runs). Summary:
[`results/summary.csv`](results/summary.csv). Original 2023 numbers:
[`results/original_2023.csv`](results/original_2023.csv).

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Windows paths shown
.venv/Scripts/python bench/get_data.py            # downloads Oxford 102 Flowers (345 MB), extracts images 1-50
.venv/Scripts/python bench/run_benchmark.py       # ~5 h on a 3700X; resumable
.venv/Scripts/python bench/make_charts.py

# optional no-GIL arm: free-threaded CPython 3.14t with numpy only
python3.14t -m venv .venv-ft && .venv-ft/Scripts/pip install numpy
.venv-ft/Scripts/python bench/run_benchmark.py --configs serial_whole,threads4,threads16   # ~1.5 h
```

## Limitations and what I'd do next

- **Make each core faster before adding cores.** OpenCV's C++ SIFT runs the same image in 88 ms, about
  600× faster than this pure-Python version. Parallelism multiplies single-core speed, so a production CBIR
  system would use compiled SIFT and parallelise across *images*, which also removes the border-keypoint
  loss (−3% at 4 tiles, −8% at 16).
- **Border loss can be fixed.** If tiling is required, overlapping tiles (a "halo") restore the lost keypoints.
- **One run per configuration per image,** on a desktop that wasn't fully isolated. Per-image speedup ranges
  are in `summary.csv`.

## Repo layout

```
nms_sift/      sift.py (NMS-SIFT), tiling.py, parallel.py (tile runner), cv_compat.py (OpenCV / NumPy shim)
bench/         get_data.py, run_benchmark.py, make_charts.py
notebooks/     parallel_sift.ipynb: the full write-up, executed
results/       raw timings, summary, charts, 2023 reference numbers
archive/       2023 checkpoint notebook (outputs stripped) and slides
```

## Credits

- **NMS-SIFT implementation:** originally written by **Sairaj Jadhav** and **Dr. John Korah**, adapted from
  the open-source PythonSIFT reference implementation of Lowe's SIFT. My contributions are the tiling,
  the parallel execution, the experiments, the diagnosis and this rebuild.
- **Dataset:** M.-E. Nilsback and A. Zisserman, "Automated flower classification over a large number of
  classes," ICVGIP 2008.
- **References:**
  - D. G. Lowe, "Object recognition from local scale-invariant features," ICCV 1999.
  - J. Hosang et al., "Learning non-maximum suppression," CVPR 2017.
  - PEP 703, "Making the Global Interpreter Lock optional in CPython."
