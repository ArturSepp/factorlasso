"""
ssr_speedup.py — reproducible benchmark for the closed-form SSR identity.

Backs the speedup claim in §1 (item 4) and §2.2 of the paper: the
vectorised closed-form pooled SSR / t-statistic

    SSR_j = ||Y||_F^2 - beta_j^2 * D_j        (Eq. 4, valid-row form)

computed in a single matrix product is materially faster than a baseline
that materialises the (T, q) residual matrix per factor column to obtain
the same t-statistics.

The benchmark compares, across panel sizes (T, N, M) spanning the
multi-asset CMA regime up to a large-universe stress point:

  * ``closed_form``  — the vectorised path used by factorlasso
  * ``materialised`` — a reference loop that builds residuals per column

Both produce identical t-statistics (verified each run); only the timing
differs. Run::

    python benchmarks/ssr_speedup.py
    python benchmarks/ssr_speedup.py --repeats 20 --sizes 50,120,9 200,240,9

Reported speedup is the ratio of median materialised time to median
closed-form time. On the panel sizes typical of multi-asset CMA
estimation the ratio spans roughly one to two orders of magnitude.
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def _closed_form_t(x_arr: np.ndarray, y_arr: np.ndarray) -> np.ndarray:
    """Vectorised pooled slope + closed-form SSR t-statistic over all factors."""
    T, M = x_arr.shape
    q = y_arr.shape[1]
    y_sum = y_arr.sum(axis=1)
    xx = (x_arr * x_arr).sum(axis=0)            # (M,)
    slopes = (x_arr.T @ y_sum) / (q * xx)       # (M,)
    Y_ss = float((y_arr * y_arr).sum())
    df = max(q * T - q, 1)
    ssr = Y_ss - q * slopes * slopes * xx       # (M,) closed form
    sigma2 = np.maximum(ssr, 0.0) / df
    se = np.sqrt(sigma2 / (q * xx))
    return np.where(se > 0, slopes / se, 0.0)


def _materialised_t(x_arr: np.ndarray, y_arr: np.ndarray) -> np.ndarray:
    """Reference: materialise the (T, q) residual matrix per factor column."""
    T, M = x_arr.shape
    q = y_arr.shape[1]
    y_sum = y_arr.sum(axis=1)
    xx = (x_arr * x_arr).sum(axis=0)
    slopes = (x_arr.T @ y_sum) / (q * xx)
    df = max(q * T - q, 1)
    t_stats = np.zeros(M)
    for j in range(M):
        xj = x_arr[:, j:j + 1]                  # (T, 1)
        resid = y_arr - slopes[j] * xj          # (T, q) — the allocation we avoid
        ssr = float((resid * resid).sum())
        sigma2 = max(ssr, 0.0) / df
        se = np.sqrt(sigma2 / (q * xx[j]))
        t_stats[j] = slopes[j] / se if se > 0 else 0.0
    return t_stats


def _median_time(fn, x, y, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(x, y)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def bench(sizes: list[tuple[int, int, int]], repeats: int) -> None:
    rng = np.random.default_rng(0)
    print(f"{'T':>5} {'N':>5} {'M':>4}  "
          f"{'closed (ms)':>12} {'materialised (ms)':>18} {'speedup':>9}")
    print("-" * 60)
    for T, N, M in sizes:
        X = rng.standard_normal((T, M))
        beta = rng.standard_normal((N, M)) * 0.4
        Y = X @ beta.T + 0.3 * rng.standard_normal((T, N))

        t_cf = _closed_form_t(X, Y)
        t_mat = _materialised_t(X, Y)
        assert np.allclose(t_cf, t_mat, atol=1e-8), "t-statistics disagree!"

        cf = _median_time(_closed_form_t, X, Y, repeats)
        mat = _median_time(_materialised_t, X, Y, repeats)
        print(f"{T:>5} {N:>5} {M:>4}  {cf*1e3:>12.4f} {mat*1e3:>18.4f} "
              f"{mat / cf:>8.1f}x")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SSR closed-form speedup benchmark.")
    parser.add_argument(
        "--repeats", type=int, default=15,
        help="Timing repeats per cell (median reported).",
    )
    parser.add_argument(
        "--sizes", nargs="*", default=None,
        help="Panel sizes as T,N,M triples (e.g. 50,120,9). Defaults to a "
        "sweep across the multi-asset CMA regime.",
    )
    args = parser.parse_args(argv)

    if args.sizes:
        sizes = [tuple(int(v) for v in s.split(",")) for s in args.sizes]
    else:
        sizes = [
            (120, 50, 9),    # base multi-asset CMA panel
            (120, 100, 9),
            (240, 200, 9),
            (378, 503, 6),   # the §6 S&P 500 application panel
            (500, 1000, 12),  # large-universe stress
        ]
    bench(sizes, args.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
