"""Scan the Engel seen_graphs.npz for record-edge-count configs at
n in {30, 40, 50, 70} using the exact scaled-integer unit test, and save
the record configs per n to runs/engel_db/engel_records_n{n}.npy."""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

RECORDS = {30: 93, 40: 137, 50: 183, 70: 281}


def batch_edge_counts(X: np.ndarray) -> np.ndarray:
    """X: (B, n, 4) int64 -> (B,) exact unit-edge counts."""
    diff = X[:, :, None, :] - X[:, None, :, :]
    da, db, dc, dd = diff[..., 0], diff[..., 1], diff[..., 2], diff[..., 3]
    A = 12 * da + 6 * db + 10 * dc + 5 * dd
    B = 6 * db + 5 * dd
    C = 2 * dc + dd
    D = -dd
    M = (A * A + 3 * B * B + 11 * C * C + 33 * D * D == 144) & (A * D + B * C == 0)
    return M.sum(axis=(1, 2)) // 2


def main() -> None:
    z = np.load(ROOT / "runs/engel_db/seen_graphs.npz", allow_pickle=False)
    for n, want in RECORDS.items():
        t0 = time.time()
        keys = sorted(
            (k for k in z.files if k.endswith(f"_{n}")),
            key=lambda k: int(k.split("_")[0]),
        )
        chunk = max(64, (1 << 24) // (n * n * 4))
        hist: Counter = Counter()
        rec_list = []
        total = 0
        best = -1
        for k in keys:
            arr = np.asarray(z[k], dtype=np.int64)
            total += len(arr)
            for i0 in range(0, len(arr), chunk):
                X = arr[i0 : i0 + chunk]
                ec = batch_edge_counts(X)
                best = max(best, int(ec.max()))
                hist.update(ec.tolist())
                hit = np.nonzero(ec == want)[0]
                if len(hit):
                    rec_list.append(X[hit].astype(np.int8))
        top = sorted(hist.items(), reverse=True)[:6]
        recs = (
            np.concatenate(rec_list) if rec_list else np.zeros((0, n, 4), np.int8)
        )
        np.save(ROOT / f"runs/engel_db/engel_records_n{n}.npy", recs)
        print(
            f"n={n}: {total:,} configs scanned in {time.time()-t0:.0f}s; "
            f"max edges {best} (record {want}); top counts {top}; "
            f"{len(recs)} record configs saved",
            flush=True,
        )


if __name__ == "__main__":
    main()
