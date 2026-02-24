"""
DMMSY - Fast Python implementation (C-compiled core via ctypes)

Compile + run:
    python dmmsy.py

The script auto-compiles dmmsy_core.c -> libdmmsy.so on first run.
"""

import ctypes
import math
import os
import pathlib
import random
import subprocess
import sys
import time
import csv

import numpy as np

# ---------------------------------------------------------------------------
# Build the shared library
# ---------------------------------------------------------------------------

_HERE = pathlib.Path(__file__).resolve().parent
_C_SRC = _HERE / "dmmsy_core.c"
_SO_PATH = _HERE / "libdmmsy.so"


def _build():
    cmd = [
        "gcc", "-O3", "-march=native", "-shared", "-fPIC",
        "-ffast-math",
        "-o", str(_SO_PATH), str(_C_SRC), "-lm",
    ]
    print(f"Compiling: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    print("Compilation successful.")


def _load_lib():
    if not _SO_PATH.exists():
        _build()
    if _C_SRC.stat().st_mtime > _SO_PATH.stat().st_mtime:
        _build()

    lib = ctypes.CDLL(str(_SO_PATH))

    # void dijkstra_ref(uint32 n, uint32* offset, uint32* edge_v, double* edge_w,
    #                   uint32 src, double* d_out, uint32* pr_out)
    lib.dijkstra_ref.restype = None
    lib.dijkstra_ref.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint32),
    ]

    # void ssp_duan_opt(uint32 n, uint32* offset, uint32* edge_v, double* edge_w,
    #                   double mean_weight, uint32 src, double* d_out, uint32* pr_out)
    lib.ssp_duan_opt.restype = None
    lib.ssp_duan_opt.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double, ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint32),
    ]

    lib.ssp_duan_res.restype = None
    lib.ssp_duan_res.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double, ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint32),
    ]

    # dijkstra_fast (reused workspace, fair comparison)
    lib.dijkstra_fast.restype = None
    lib.dijkstra_fast.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint32),
    ]

    return lib


_lib = _load_lib()


# ---------------------------------------------------------------------------
# Numpy helper to get ctypes pointer from array
# ---------------------------------------------------------------------------

def _u32p(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))

def _f64p(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

_NULL_U32 = ctypes.POINTER(ctypes.c_uint32)()
_NULL_F64 = ctypes.POINTER(ctypes.c_double)()


# ---------------------------------------------------------------------------
# CSR Graph (numpy arrays)
# ---------------------------------------------------------------------------

class CSRGraph:
    __slots__ = ("n", "m", "offset", "edge_v", "edge_w", "mean_weight",
                 "_off_p", "_ev_p", "_ew_p")

    def __init__(self, n, m, offset, edge_v, edge_w, mean_weight):
        self.n = np.uint32(n)
        self.m = np.uint32(m)
        self.offset = np.ascontiguousarray(offset, dtype=np.uint32)
        self.edge_v = np.ascontiguousarray(edge_v, dtype=np.uint32)
        self.edge_w = np.ascontiguousarray(edge_w, dtype=np.float64)
        self.mean_weight = float(mean_weight)
        # Cache ctypes pointers (avoids repeated conversion)
        self._off_p = _u32p(self.offset)
        self._ev_p = _u32p(self.edge_v)
        self._ew_p = _f64p(self.edge_w)


def random_graph(n: int, m: int, max_w: float = 100.0) -> CSRGraph:
    """Fully vectorised random graph generation."""
    us = np.random.randint(0, n, size=m, dtype=np.uint32)
    vs = np.random.randint(0, n, size=m, dtype=np.uint32)
    ws = np.random.random(m) * max_w

    order = np.argsort(us, kind="stable")
    us_sorted = us[order]

    counts = np.bincount(us_sorted, minlength=n).astype(np.uint32)
    offset = np.zeros(n + 1, dtype=np.uint32)
    np.cumsum(counts, out=offset[1:])

    mean_weight = float(ws.sum() / m) if m > 0 else 0.0
    return CSRGraph(n, m, offset, vs[order].astype(np.uint32),
                    ws[order].astype(np.float64), mean_weight)


# ---------------------------------------------------------------------------
# Algorithm wrappers
# ---------------------------------------------------------------------------

def dijkstra_ref(g: CSRGraph, src: int):
    d = np.empty(g.n, dtype=np.float64)
    pr = np.empty(g.n, dtype=np.uint32)
    _lib.dijkstra_ref(g.n, g._off_p, g._ev_p, g._ew_p,
                      np.uint32(src), _f64p(d), _u32p(pr))
    return d, pr


def ssp_duan(g: CSRGraph, src: int):
    d = np.empty(g.n, dtype=np.float64)
    pr = np.empty(g.n, dtype=np.uint32)
    _lib.ssp_duan_opt(g.n, g._off_p, g._ev_p, g._ew_p,
                      g.mean_weight, np.uint32(src), _f64p(d), _u32p(pr))
    return d, pr


def ssp_duan_research(g: CSRGraph, src: int):
    d = np.empty(g.n, dtype=np.float64)
    pr = np.empty(g.n, dtype=np.uint32)
    _lib.ssp_duan_res(g.n, g._off_p, g._ev_p, g._ew_p,
                      g.mean_weight, np.uint32(src), _f64p(d), _u32p(pr))
    return d, pr


def _dijkstra_noalloc(g, src, d_buf, pr_buf):
    _lib.dijkstra_ref(g.n, g._off_p, g._ev_p, g._ew_p,
                      np.uint32(src), _f64p(d_buf), _u32p(pr_buf))

def _dijkstra_fast_noalloc(g, src, d_buf, pr_buf):
    _lib.dijkstra_fast(g.n, g._off_p, g._ev_p, g._ew_p,
                       np.uint32(src), _f64p(d_buf), _u32p(pr_buf))

def _opt_noalloc(g, src, d_buf, pr_buf):
    _lib.ssp_duan_opt(g.n, g._off_p, g._ev_p, g._ew_p,
                      g.mean_weight, np.uint32(src), _f64p(d_buf), _u32p(pr_buf))

def _res_noalloc(g, src, d_buf, pr_buf):
    _lib.ssp_duan_res(g.n, g._off_p, g._ev_p, g._ew_p,
                      g.mean_weight, np.uint32(src), _f64p(d_buf), _u32p(pr_buf))


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def assert_arrays_approx_equal(a, b, tol=1e-6):
    inf_a, inf_b = np.isinf(a), np.isinf(b)
    if not np.array_equal(inf_a, inf_b):
        return False
    finite = ~inf_a
    return np.allclose(a[finite], b[finite], atol=tol)


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

def run_comprehensive_tests():
    print("Running SSSP Correctness Tests...")

    # Diamond graph
    g = CSRGraph(
        n=4, m=4,
        offset=np.array([0, 2, 3, 4, 4], dtype=np.uint32),
        edge_v=np.array([1, 2, 3, 3], dtype=np.uint32),
        edge_w=np.array([2.0, 3.0, 1.0, 1.0]),
        mean_weight=7.0 / 4.0,
    )
    d_ref, _ = dijkstra_ref(g, 0)
    d_opt, _ = ssp_duan(g, 0)
    d_res, _ = ssp_duan_research(g, 0)
    assert assert_arrays_approx_equal(d_ref, d_opt), "Diamond failed (opt)"
    assert assert_arrays_approx_equal(d_ref, d_res), "Diamond failed (res)"

    for n, m, label in [(1000, 5000, "1k"), (10000, 50000, "10k"),
                        (100000, 500000, "100k")]:
        np.random.seed(42)
        g = random_graph(n, m)
        d_ref, _ = dijkstra_ref(g, 0)
        d_opt, _ = ssp_duan(g, 0)
        d_res, _ = ssp_duan_research(g, 0)
        # Also verify dijkstra_fast
        d_fast = np.empty(g.n, dtype=np.float64)
        pr_fast = np.empty(g.n, dtype=np.uint32)
        _lib.dijkstra_fast(g.n, g._off_p, g._ev_p, g._ew_p,
                           np.uint32(0), _f64p(d_fast), _u32p(pr_fast))
        assert assert_arrays_approx_equal(d_ref, d_opt), f"{label} failed (opt)"
        assert assert_arrays_approx_equal(d_ref, d_res), f"{label} failed (res)"
        assert assert_arrays_approx_equal(d_ref, d_fast), f"{label} failed (dij_fast)"

    print("All correctness checks PASSED.")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark():
    np.random.seed(1234)

    test_cases = [
        (1000,    5000,    100),
        (5000,    25000,   100),
        (10000,   50000,   50),
        (25000,   125000,  30),
        (50000,   250000,  20),
        (75000,   375000,  15),
        (100000,  500000,  15),
        (150000,  750000,  10),
        (200000,  1000000, 8),
        (250000,  1250000, 10),
        (350000,  1750000, 6),
        (500000,  2500000, 20),
        (750000,  3750000, 15),
        (1000000, 5000000, 10),
    ]

    print("DMMSY Performance Reporter (Python + C compiled core via ctypes)")
    run_comprehensive_tests()

    sep = "=" * 95
    print(sep)
    print(f"{'n':<10} {'m':<10} {'Dij(alloc)':<12} {'Dij(fast)':<12} {'DMMSY Res':<12} {'DMMSY Opt':<12} {'Spd(fair)':<10}")
    print("-" * 95)

    csv_rows = []

    for n, m, trials in test_cases:
        g = random_graph(n, m, 100.0)
        d_buf = np.empty(n, dtype=np.float64)
        pr_buf = np.empty(n, dtype=np.uint32)

        # Warmup
        nw = min(max(trials // 2, 1), 5)
        for _ in range(nw):
            _dijkstra_noalloc(g, 0, d_buf, pr_buf)
            _dijkstra_fast_noalloc(g, 0, d_buf, pr_buf)
            _opt_noalloc(g, 0, d_buf, pr_buf)
            _res_noalloc(g, 0, d_buf, pr_buf)

        # Dijkstra (alloc each call)
        t0 = time.perf_counter()
        for _ in range(trials):
            _dijkstra_noalloc(g, 0, d_buf, pr_buf)
        t1 = time.perf_counter()
        avg_dij = ((t1 - t0) / trials) * 1000.0

        # Dijkstra fast (reused workspace)
        t0 = time.perf_counter()
        for _ in range(trials):
            _dijkstra_fast_noalloc(g, 0, d_buf, pr_buf)
        t1 = time.perf_counter()
        avg_dij_fast = ((t1 - t0) / trials) * 1000.0

        # DMMSY Opt
        t0 = time.perf_counter()
        for _ in range(trials):
            _opt_noalloc(g, 0, d_buf, pr_buf)
        t1 = time.perf_counter()
        avg_opt = ((t1 - t0) / trials) * 1000.0

        # DMMSY Res
        t0 = time.perf_counter()
        for _ in range(trials):
            _res_noalloc(g, 0, d_buf, pr_buf)
        t1 = time.perf_counter()
        avg_res = ((t1 - t0) / trials) * 1000.0

        spd_fair = avg_dij_fast / avg_opt if avg_opt > 0 else float("inf")
        print(f"{n:<10} {m:<10} {avg_dij:<12.4f} {avg_dij_fast:<12.4f} {avg_res:<12.4f} {avg_opt:<12.4f} {spd_fair:<10.2f}x")
        csv_rows.append({"n": n, "m": m, "dijkstra": avg_dij,
                         "dijkstra_fast": avg_dij_fast,
                         "dmmsy_res": avg_res, "dmmsy_opt": avg_opt})

    print(sep)
    print("Full benchmark complete.")

    with open("benchmark_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n", "m", "dijkstra", "dijkstra_fast",
                                          "dmmsy_res", "dmmsy_opt"])
        w.writeheader()
        for row in csv_rows:
            w.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in row.items()})

    with open("benchmark_data.js", "w") as js_f:
        js_f.write("const BENCHMARK_CSV_DATA = `\\n")
        with open("benchmark_data.csv") as csv_f:
            js_f.write(csv_f.read())
        js_f.write("`;\n")

    print("Results saved to benchmark_data.csv and benchmark_data.js")


if __name__ == "__main__":
    run_benchmark()
