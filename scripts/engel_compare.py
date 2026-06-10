"""Gather our record-tying configs, class them, and compare against the
Engel-Hammond-Lee-Su-Varga-Zsamboki FigShare DB (seen_graphs.npz).

Levels of comparison:
1. Lattice congruence: chase40lib.canon() -- the 12 ML rigid motions
   (6 rotations by w1^k x optional reflection z -> w3*conj(z)) + translation.
   canon-equal  <=>  identical point configuration up to ML symmetry.
2. Graph isomorphism: edge sets compared by WL-refinement fingerprint, with
   exact VF2 confirmation done separately (networkx, via uv run --with).

Usage:
  uv run python scripts/engel_compare.py gather      # our classes only
  uv run python scripts/engel_compare.py compare     # needs runs/engel_db/seen_graphs.npz
"""

from __future__ import annotations

import glob
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from chase40lib import canon, edge_count, edges_of  # noqa: E402

RECORDS = {30: 93, 40: 137, 50: 183, 70: 281}

OUR_GLOBS = {
    30: [
        "data/mlcoords/udg30_93edges*.json",
        "runs/chase/n30/*93edges*.json",
    ],
    40: [
        "data/mlcoords/udg40_137edges*.json",
        "runs/chase/n40/**/*137*.json",
    ],
    50: [
        "data/mlcoords/udg50_183edges*.json",
        "runs/chase/n50/mlcoords/udg50_183edges*.json",
        "runs/chase/n50/pool_*_183e_*.json",
        "runs/chase/n50/udg50_183edges_*.json",
    ],
    70: [
        "data/mlcoords/udg70_281edges*.json",
        "runs/chase/n70/mlcerts/anneal_281_*.json",
        "runs/chase/n70/final/udg70_281edges*_exact.json",
    ],
}


def load_coords(path: Path) -> np.ndarray | None:
    try:
        with open(path) as f:
            d = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    coords = d.get("coords") if isinstance(d, dict) else d
    if coords is None:
        return None
    arr = np.asarray(coords)
    if arr.ndim != 2 or arr.shape[1] != 4:
        return None
    return arr.astype(np.int64)


def wl_fingerprint(n: int, edges: list[tuple[int, int]], rounds: int = 4) -> str:
    """Weisfeiler-Leman colour-refinement hash of the graph (iso invariant)."""
    adj: list[list[int]] = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    col = [len(a) for a in adj]
    for _ in range(rounds):
        sigs = [hash((col[v], tuple(sorted(col[u] for u in adj[v])))) for v in range(n)]
        remap = {s: k for k, s in enumerate(sorted(set(sigs)))}
        col = [remap[s] for s in sigs]
    hist = tuple(sorted(Counter(col).items()))
    return hashlib.sha256(repr((n, len(edges), hist)).encode()).hexdigest()[:16]


def gather_ours() -> dict[int, dict]:
    out: dict[int, dict] = {}
    for n, want in RECORDS.items():
        files: list[Path] = []
        for g in OUR_GLOBS[n]:
            files.extend(Path(p) for p in glob.glob(str(ROOT / g), recursive=True))
        files = sorted(set(files))
        classes: dict[bytes, dict] = {}
        skipped = []
        for f in files:
            P = load_coords(f)
            if P is None or len(P) != n:
                skipped.append((str(f.relative_to(ROOT)), "not an n-point coords json"))
                continue
            ec = edge_count(P)
            if ec != want:
                skipped.append((str(f.relative_to(ROOT)), f"{ec} edges"))
                continue
            key = canon(P)
            if key not in classes:
                classes[key] = {
                    "rep": str(f.relative_to(ROOT)),
                    "coords": P,
                    "wl": wl_fingerprint(n, edges_of(P)),
                    "hits": 0,
                }
            classes[key]["hits"] += 1
        out[n] = {"classes": classes, "skipped": skipped, "nfiles": len(files)}
    return out


def report_ours(ours: dict[int, dict]) -> None:
    for n, want in RECORDS.items():
        info = ours[n]
        cl = info["classes"]
        wl_groups = Counter(c["wl"] for c in cl.values())
        print(f"\nn={n} ({want} edges): {info['nfiles']} files scanned, "
              f"{sum(c['hits'] for c in cl.values())} record configs, "
              f"{len(cl)} lattice-congruence classes, "
              f"{len(wl_groups)} WL graph-fingerprint groups")
        for k, c in sorted(cl.items(), key=lambda kv: kv[1]["rep"]):
            print(f"  [{c['wl']}] x{c['hits']:<3} {c['rep']}")
        for path, why in info["skipped"][:6]:
            print(f"    (skipped {path}: {why})")
        if len(info["skipped"]) > 6:
            print(f"    (... {len(info['skipped']) - 6} more skipped)")


def load_engel() -> dict[int, np.ndarray]:
    """Record-edge-count configs extracted from the npz by engel_scan.py."""
    out = {}
    for n in RECORDS:
        p = ROOT / f"runs/engel_db/engel_records_n{n}.npy"
        if p.exists():
            out[n] = np.load(p)
    return out


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "gather"
    ours = gather_ours()
    report_ours(ours)
    if mode != "compare":
        return

    print("\n=== Engel DB record configs (from engel_scan.py) ===")
    engel = load_engel()
    summary = {}
    for n, want in RECORDS.items():
        arr = engel.get(n)
        if arr is None:
            print(f"n={n}: engel_records_n{n}.npy missing — run engel_scan.py")
            continue
        arr = arr.astype(np.int64)
        m = len(arr)
        print(f"\nn={n}: {m} record-count ({want}-edge) configs in DB")
        their_classes: dict[bytes, dict] = {}
        for i in range(m):
            P = arr[i]
            key = canon(P)
            if key not in their_classes:
                their_classes[key] = {
                    "idx": int(i),
                    "wl": wl_fingerprint(n, edges_of(P)),
                    "hits": 0,
                    "coords": P,
                }
            their_classes[key]["hits"] += 1
        print(f"  their lattice-congruence classes: {len(their_classes)}, "
              f"WL groups: {len(set(c['wl'] for c in their_classes.values()))}")

        our_cl = ours[n]["classes"]
        their_keys = set(their_classes)
        their_wls = {c["wl"] for c in their_classes.values()}
        rows = []
        for k, c in sorted(our_cl.items(), key=lambda kv: kv[1]["rep"]):
            lat = "LATTICE-IDENTICAL" if k in their_keys else "lattice-distinct"
            wl = "WL-match" if c["wl"] in their_wls else "WL-NEW"
            rows.append((c["rep"], lat, wl))
            print(f"    {c['rep']}: {lat}, {wl}")
        summary[n] = {
            "ours": len(our_cl),
            "theirs_lat": len(their_classes),
            "theirs_wl": len(their_wls),
            "rows": rows,
        }
        # dump reps for the exact-iso pass
        dump = {
            "n": n,
            "ours": [
                {"rep": c["rep"], "wl": c["wl"],
                 "edges": edges_of(c["coords"]),
                 "lattice_identical": (k in their_keys)}
                for k, c in sorted(our_cl.items(), key=lambda kv: kv[1]["rep"])
            ],
            "theirs": [
                {"idx": c["idx"], "wl": c["wl"], "edges": edges_of(c["coords"])}
                for c in sorted(their_classes.values(), key=lambda c: c["idx"])
            ],
        }
        with open(ROOT / f"runs/engel_db/classes_n{n}.json", "w") as f:
            json.dump(dump, f)
    print("\nwrote runs/engel_db/classes_n*.json for the exact-iso pass")


if __name__ == "__main__":
    main()
