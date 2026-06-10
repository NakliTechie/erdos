"""Scan the FULL Engel seen_graphs.npz: per-n max exact edge count over every
config in the DB (= the densest-known record at that n, since the DB contains
the searches' seen graphs including all record configs). Also saves, per n, the
configs achieving the max (the record inventory) for instant novelty diffs.

Output: runs/engel_db/targets.json {n: {"target": max_edges, "n_configs": total,
"n_record_configs": count}} + runs/engel_db/engel_records_n{n}.npy for every n.

Parallel over n (each n's batch arrays are independent).
Usage: uv run python scripts/engel_targets.py [--processes 10]
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NPZ = ROOT / "runs/engel_db/seen_graphs.npz"


def batch_edge_counts(X: np.ndarray) -> np.ndarray:
    """X: (B, n, 4) int64 -> (B,) exact unit-edge counts (scaled-integer test)."""
    diff = X[:, :, None, :] - X[:, None, :, :]
    da, db, dc, dd = diff[..., 0], diff[..., 1], diff[..., 2], diff[..., 3]
    A = 12 * da + 6 * db + 10 * dc + 5 * dd
    B = 6 * db + 5 * dd
    C = 2 * dc + dd
    D = -dd
    M = (A * A + 3 * B * B + 11 * C * C + 33 * D * D == 144) & (A * D + B * C == 0)
    return M.sum(axis=(1, 2)) // 2


def scan_n(n: int) -> tuple[int, int, int, int]:
    z = np.load(NPZ, allow_pickle=False)
    keys = sorted(
        (k for k in z.files if k.endswith(f"_{n}")),
        key=lambda k: int(k.split("_")[0]),
    )
    chunk = max(64, (1 << 24) // (n * n * 4))
    total = 0
    best = -1
    rec_list: list[np.ndarray] = []
    for k in keys:
        arr = np.asarray(z[k], dtype=np.int64)
        total += len(arr)
        for i0 in range(0, len(arr), chunk):
            X = arr[i0 : i0 + chunk]
            ec = batch_edge_counts(X)
            m = int(ec.max())
            if m > best:
                best = m
                rec_list = []
            if m >= best:
                hit = np.nonzero(ec == best)[0]
                if len(hit):
                    rec_list.append(X[hit].astype(np.int8))
    recs = np.concatenate(rec_list) if rec_list else np.zeros((0, n, 4), np.int8)
    out = ROOT / f"runs/engel_db/engel_records_n{n}.npy"
    if not out.exists():  # don't clobber the session-2 extracts
        np.save(out, recs)
    return n, best, total, len(recs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processes", type=int, default=10)
    args = ap.parse_args()

    z = np.load(NPZ, allow_pickle=False)
    ns = sorted({int(k.split("_")[1]) for k in z.files})
    del z
    print(f"{len(ns)} distinct n in DB: {ns[0]}..{ns[-1]}", flush=True)

    t0 = time.time()
    targets: dict[int, dict] = {}
    with ProcessPoolExecutor(max_workers=args.processes) as ex:
        for n, best, total, nrec in ex.map(scan_n, ns):
            targets[n] = {"target": best, "n_configs": total, "n_record_configs": nrec}
            print(
                f"n={n}: max {best} over {total:,} configs ({nrec} at max) "
                f"[{time.time()-t0:.0f}s elapsed]",
                flush=True,
            )
    (ROOT / "runs/engel_db/targets.json").write_text(
        json.dumps({str(k): targets[k] for k in sorted(targets)}, indent=1)
    )
    print(f"targets.json written: {len(targets)} n values in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
