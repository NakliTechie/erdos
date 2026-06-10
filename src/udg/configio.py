"""Config I/O for the udg package.

CSV format (matches data/udg40_132edges.csv):
    - first line is the literal header ``x,y`` (no comment prefix)
    - one point per row, comma-delimited, scientific notation with full
      float64 precision so save -> load round-trips EXACTLY (bit-for-bit).

Also: dataclass -> JSON for audit reports (numpy scalars cast to python
types) and a tiny append helper for the running RESULTS.md log.

Self-contained: stdlib + numpy only, no intra-package imports.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import numpy as np

# np.savetxt default precision; 18 fractional digits in %e is more than the
# 17 significant digits float64 needs, so round-trip is exact. This is the
# exact format of data/udg40_132edges.csv.
_FMT = "%.18e"
_HEADER = "x,y"


def load_csv(path: str | os.PathLike) -> np.ndarray:
    """Read a config CSV (header ``x,y``) -> (n, 2) float64 array."""
    P = np.loadtxt(Path(path), dtype=np.float64, delimiter=",", skiprows=1)
    P = np.atleast_2d(P)  # single-row file comes back as shape (2,)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError(f"{path}: expected 2 columns (x,y), got shape {P.shape}")
    return P


def save_csv(P: np.ndarray, path: str | os.PathLike) -> None:
    """Write points as CSV with header ``x,y``, full float64 precision.

    Round-trips exactly: ``load_csv(p)`` after ``save_csv(P, p)`` returns an
    array equal to ``P`` with atol=0, rtol=0.
    """
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError(f"expected (n, 2) array, got shape {P.shape}")
    np.savetxt(Path(path), P, fmt=_FMT, delimiter=",", header=_HEADER, comments="")


def _jsonable(v):
    """Recursively cast numpy scalars/arrays to plain python types."""
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def save_audit_json(report, path: str | os.PathLike) -> None:
    """Serialize an AuditReport (or any dataclass instance / mapping) to JSON.

    Numpy scalars and arrays are cast to plain python types so the file is
    readable by any JSON consumer.
    """
    if dataclasses.is_dataclass(report) and not isinstance(report, type):
        d = dataclasses.asdict(report)
    elif isinstance(report, dict):
        d = dict(report)
    else:
        raise TypeError(f"expected a dataclass instance or dict, got {type(report)}")
    with open(Path(path), "w", encoding="utf-8") as f:
        json.dump(_jsonable(d), f, indent=2)
        f.write("\n")


def append_results_line(path: str | os.PathLike, line: str) -> None:
    """Append one markdown line to RESULTS.md (created if missing)."""
    with open(Path(path), "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")
