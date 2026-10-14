"""The handful of OpenCV primitives NMS-SIFT needs, with a pure-NumPy fallback.

OpenCV ships no wheels for free-threaded (no-GIL) CPython yet, so on a 3.14t
interpreter we fall back to NumPy re-implementations of GaussianBlur, resize,
subtract and KeyPoint. They follow OpenCV's conventions (kernel size rule,
BORDER_REFLECT_101, half-pixel bilinear sampling, float32 KeyPoint fields). They
agree to float rounding, not bit-for-bit, so keypoint counts can differ by a few.
These calls are a negligible share of runtime; the hot loops are pure Python.
"""
from __future__ import annotations

import os

import numpy as np

_FORCE_NUMPY = os.environ.get("NMS_SIFT_NUMPY_BACKEND") == "1"

try:
    if _FORCE_NUMPY:
        raise ImportError
    import cv2

    BACKEND = "opencv"
    GaussianBlur = cv2.GaussianBlur
    resize = cv2.resize
    subtract = cv2.subtract
    KeyPoint = cv2.KeyPoint
    INTER_LINEAR = cv2.INTER_LINEAR
    INTER_NEAREST = cv2.INTER_NEAREST

    def imread_gray(path: str) -> np.ndarray:
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

except ImportError:
    BACKEND = "numpy"
    INTER_LINEAR, INTER_NEAREST = 1, 0

    def _gaussian_kernel(sigma: float) -> np.ndarray:
        ksize = int(np.round(sigma * 4 * 2 + 1)) | 1  # OpenCV rule for float images
        x = np.arange(ksize, dtype=np.float64) - (ksize - 1) / 2
        k = np.exp(-(x ** 2) / (2 * sigma ** 2))
        return (k / k.sum()).astype(np.float32)

    def _filter_axis(img: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
        r = len(k) // 2
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        p = np.pad(img, pad, mode="reflect")  # == BORDER_REFLECT_101
        n = img.shape[axis]
        out = np.zeros_like(img, dtype=np.float32)
        for t, w in enumerate(k):
            out += w * (p[t:t + n, :] if axis == 0 else p[:, t:t + n])
        return out

    def GaussianBlur(img, ksize, sigmaX, sigmaY=None):
        img = np.asarray(img, dtype=np.float32)
        kx = _gaussian_kernel(float(sigmaX))
        ky = _gaussian_kernel(float(sigmaY if sigmaY else sigmaX))
        return _filter_axis(_filter_axis(img, kx, 1), ky, 0)

    def _linear_axis(img: np.ndarray, dst_n: int, axis: int) -> np.ndarray:
        src_n = img.shape[axis]
        pos = (np.arange(dst_n) + 0.5) * (src_n / dst_n) - 0.5
        i0 = np.floor(pos).astype(int)
        frac = (pos - i0).astype(np.float32)
        lo = np.clip(i0, 0, src_n - 1)
        hi = np.clip(i0 + 1, 0, src_n - 1)
        a, b = np.take(img, lo, axis=axis), np.take(img, hi, axis=axis)
        f = frac[:, None] if axis == 0 else frac[None, :]
        return a * (1 - f) + b * f

    def resize(img, dsize, fx=0, fy=0, interpolation=INTER_LINEAR):
        h, w = img.shape
        dw, dh = dsize if dsize and dsize != (0, 0) else (int(round(w * fx)), int(round(h * fy)))
        if interpolation == INTER_NEAREST:
            ry = np.minimum(np.floor(np.arange(dh) * (h / dh)).astype(int), h - 1)
            rx = np.minimum(np.floor(np.arange(dw) * (w / dw)).astype(int), w - 1)
            return np.ascontiguousarray(img[ry][:, rx])
        img = np.asarray(img, dtype=np.float32)
        return _linear_axis(_linear_axis(img, dw, 1), dh, 0).astype(np.float32)

    def subtract(a, b):
        return np.subtract(a, b, dtype=np.float32)

    class KeyPoint:
        """Minimal cv2.KeyPoint stand-in; fields are float32-rounded like OpenCV's."""
        __slots__ = ("_pt", "_size", "_angle", "_response", "octave", "class_id")

        def __init__(self, x=0.0, y=0.0, size=0.0, angle=-1.0, response=0.0, octave=0, class_id=-1):
            self.pt = (x, y)
            self.size, self.angle, self.response = size, angle, response
            self.octave, self.class_id = int(octave), int(class_id)

        f32 = staticmethod(lambda v: float(np.float32(v)))
        pt = property(lambda s: s._pt, lambda s, v: setattr(s, "_pt", (KeyPoint.f32(v[0]), KeyPoint.f32(v[1]))))
        size = property(lambda s: s._size, lambda s, v: setattr(s, "_size", KeyPoint.f32(v)))
        angle = property(lambda s: s._angle, lambda s, v: setattr(s, "_angle", KeyPoint.f32(v)))
        response = property(lambda s: s._response, lambda s, v: setattr(s, "_response", KeyPoint.f32(v)))

    def imread_gray(path: str) -> np.ndarray:
        raise RuntimeError("Image decoding needs OpenCV; pass pre-decoded arrays (see bench/run_benchmark.py)")
