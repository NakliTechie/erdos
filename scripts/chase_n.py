#!/usr/bin/env python
"""Generic per-n record-chase driver (the Batch-F frontier-sweep engine).

Escalating attack ladder over exact Moser-lattice coordinates, stopping
early the moment an AUDIT-PASSING best >= target (a raw edge count that
fails the float three-audit never stops a stage, never touches the stop
file, and never backs a tied/exceeded verdict -- the n=71/72 band-B4
regression: subset ties from t3xt3c1/t2xt4c1 closure ambients with an
inherent 0.0851 < MIN_SEP pair):

  (a) subset:  candidate ambients near n (wheel49, Eisenstein x w3 patch
               cross-sums, their 1-step closures) + densest-n-subgraph ILS
               (udg.subsetsearch -- the n=30/93 engine);
  (b) beam:    diverse beam search over exact coords, width ladder
               300 -> 700 -> 1500 (the chase40_beam recipe, any n);
  (c) anneal:  exact-lattice simulated-annealing pool seeded from the
               stage bests (udg.anneal -- the n=70/281 engine);
  (d) polish:  float-side exploit loop (warm Metropolis + coincidence
               forcing), then exact re-certification.

EVERY claimed best passes the float three-audit (udg.audit) AND exact
certification in Q(sqrt3, sqrt11) (scripts/ml_coords.py). Outputs in --out:
CSV + exact-coords JSON + audit JSON + certificate JSON per claimed best,
and summary.json with per-stage bests and timings. If best > target, a
LOUD "NEW RECORD CANDIDATE" line is printed and the config is diffed
against the local Engel DB (runs/engel_db/seen_graphs.npz, exact batch
edge counts + udg.mlgraph.canon lattice identity).

Target resolution: --target, else runs/engel_db/targets.json, else the
edge count of runs/engel_db/engel_records_n<n>.npy.

Determinism: all stage/task seeds derive from --seed; each completed task
is reproducible -- the wall-clock budget only decides how many tasks run.

Usage:
  uv run python scripts/chase_n.py --n 30 --target 93 --budget-min 12 \
      --out runs/sweep/n30 --processes 6 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from udg.anneal import anneal_pool, greedy_add  # noqa: E402
from udg.audit import audit  # noqa: E402
from udg.configio import load_csv, save_audit_json, save_csv  # noqa: E402
from udg.mlgraph import MLConfig, canon, minkowski, to_float, tri_patch  # noqa: E402
from udg.search import multi_search  # noqa: E402
from udg.subsetsearch import (  # noqa: E402
    edge_count,
    hex_patch,
    neighbor_closure,
    subset_ils,
    to_w3,
    wheel49,
)

from chase40_beam import beam_run, full_polish  # noqa: E402
from chase40lib import sep_viol_mask, unit_mask as c40_unit_mask, universe  # noqa: E402
from engel_targets import batch_edge_counts  # noqa: E402
from force_coincidences import force  # noqa: E402
from ml_coords import certify_config, sanity_check_arithmetic  # noqa: E402

STAGE_WEIGHTS = {"subset": 0.20, "beam": 0.35, "anneal": 0.30, "polish": 0.15}
NPZ = ROOT / "runs/engel_db/seen_graphs.npz"


def log(msg: str) -> None:
    print(f"[chase_n] {msg}", flush=True)


# ---------------------------------------------------------------------------
# the audit gate: a raw edge count is not a claim (HANDOFF §3)
# ---------------------------------------------------------------------------

def audit_passes(pts) -> bool:
    """Float three-audit (udg.audit) on exact lattice points.

    The gate every raw target hit must clear before it may stop a stage,
    touch the stop file, or back a tied/exceeded verdict. Subset configs
    inherit the ambient's geometry, and closure ambients can contain point
    pairs below MIN_SEP (n=71/72: 0.0851 in t3xt3c1/t2xt4c1)."""
    cfg = MLConfig(
        tuple(int(x) for x in p) for p in np.asarray(pts, dtype=np.int64)
    )
    return bool(audit(to_float(cfg)).passed)


def ladder_done(global_best, target: int) -> bool:
    """May the stage ladder stop early? Only on an audit-passing best
    >= target; an audit-failing raw tie must keep the ladder running."""
    if global_best is None:
        return False
    return bool(global_best[4] and global_best[0] >= target)


def final_verdict(edges: int, target: int, info: dict) -> dict:
    """tied/exceeded are CLAIMS: beyond the raw count they require the
    float three-audit AND exact certification from the final claim info."""
    valid = bool(info.get("audit_passed")) and bool(info.get("certified"))
    return {
        "tied": bool(valid and edges == target),
        "exceeded": bool(valid and edges > target),
    }


# ---------------------------------------------------------------------------
# target resolution
# ---------------------------------------------------------------------------

def resolve_target(n: int, cli_target: int | None) -> tuple[int, str]:
    if cli_target is not None:
        return cli_target, "--target"
    tj = ROOT / "runs/engel_db/targets.json"
    if tj.exists():
        d = json.loads(tj.read_text())
        if str(n) in d:
            return int(d[str(n)]["target"]), "runs/engel_db/targets.json"
    rec = ROOT / f"runs/engel_db/engel_records_n{n}.npy"
    if rec.exists():
        arr = np.load(rec).astype(np.int64)
        if len(arr):
            m = int(batch_edge_counts(arr[:8]).max())
            return m, f"runs/engel_db/engel_records_n{n}.npy"
    raise SystemExit(
        f"no target for n={n}: pass --target, or generate "
        "runs/engel_db/targets.json with scripts/engel_targets.py"
    )


# ---------------------------------------------------------------------------
# stage (a): subset-in-closure
# ---------------------------------------------------------------------------

# Eisenstein-sublattice piece menu (name -> builder); cross sums
# A (+) w3*B are exactly |A|*|B| points (free module: no collisions).
_PIECES = {
    "t2": lambda: tri_patch(2),   # 6 pts, 9 edges (the P6 patch)
    "t3": lambda: tri_patch(3),   # 10 pts
    "t4": lambda: tri_patch(4),   # 15 pts
    "h1": lambda: hex_patch(1),   # 7 pts (6-wheel)
    "h2": lambda: hex_patch(2),   # 19 pts
    "h3": lambda: hex_patch(3),   # 37 pts
}


def build_ambients(n: int, max_cross: int = 4) -> list[tuple[str, np.ndarray]]:
    """Candidate ambient ML point sets near n: wheel49 closure-1 and
    Eisenstein x w3 patch cross-sums (with closure-1 for tight sums)."""
    ambs: list[tuple[str, np.ndarray]] = []
    if n <= 55:
        ambs.append(
            ("w49c1", neighbor_closure(wheel49().as_array().astype(np.int64), 1))
        )
    sizes = {k: len(f()) for k, f in _PIECES.items()}
    crosses = []
    for a, pa in sizes.items():
        for b, pb in sizes.items():
            sz = pa * pb
            if n <= sz <= 30 * n:
                crosses.append((sz, a, b))
    crosses.sort()
    seen_sz = set()
    picked = []
    for sz, a, b in crosses:
        if sz in seen_sz:
            continue
        seen_sz.add(sz)
        picked.append((sz, a, b))
        if len(picked) >= max_cross:
            break
    for sz, a, b in picked:
        S = minkowski(_PIECES[a](), to_w3(_PIECES[b]())).as_array().astype(np.int64)
        ambs.append((f"{a}x{b}", S))
        if sz <= 1.7 * n:
            ambs.append((f"{a}x{b}c1", neighbor_closure(S, 1)))
    return ambs


def _subset_task(payload):
    """Restart ladder of short subset_ils runs (calibration: restart
    diversity beats one long kick chain) within the wall-clock slice.

    A raw target hit ends the ladder ONLY if it passes the float audit;
    an audit-failing tie just re-loops on a fresh seed (subset_ils itself
    early-stops on the raw count, so each bad tie costs one restart)."""
    (name, amb, k, target, seed, minutes, stop_path) = payload
    amb = np.asarray(amb, dtype=np.int64)
    stop = (lambda: Path(stop_path).exists()) if stop_path else (lambda: False)
    deadline = time.time() + minutes * 60
    best_m, best_pts, best_ok = -1, None, False
    r = 0
    while time.time() < deadline and not stop():
        m, pts = subset_ils(
            amb, k, seed=seed * 100_003 + r, max_iters=150,
            minutes=max(0.02, (deadline - time.time()) / 60.0),
            target=target, should_stop=stop,
        )
        ok = m >= target and audit_passes(pts)
        # an audit-passing target hit outranks any raw count that fails it
        if (ok, m) > (best_ok, best_m):
            best_m, best_pts, best_ok = m, pts, ok
        if best_ok:
            break
        r += 1
    if best_pts is None:  # stopped before the first restart finished
        return name, -1, None, False
    return name, best_m, [tuple(int(x) for x in row) for row in best_pts], best_ok


def run_subset_stage(n, target, budget_s, processes, seed, stop_path, found):
    ambients = build_ambients(n)
    log(
        "subset ambients: "
        + ", ".join(f"{name}({len(a)})" for name, a in ambients)
    )
    if not ambients:
        return None
    deadline = time.time() + budget_s
    best = None  # (edges, pts, label); audit-passing target hits outrank raw counts
    best_ok = False
    waves = [ambients[i : i + processes] for i in range(0, len(ambients), processes)]
    for w, wave in enumerate(waves):
        remaining = deadline - time.time()
        if remaining < 5:
            break
        slice_min = (remaining / (len(waves) - w)) / 60.0
        payloads = [
            (name, amb, n, target, seed * 1_000_003 + w * 1009 + i, slice_min,
             stop_path)
            for i, (name, amb) in enumerate(wave)
        ]
        with ProcessPoolExecutor(max_workers=processes) as ex:
            futs = [ex.submit(_subset_task, p) for p in payloads]
            for fut in as_completed(futs):
                name, m, pts, ok = fut.result()
                if pts is None:
                    continue
                log(f"subset[{name}]: {m} edges"
                    + (" (target hit FAILS audit — not stopping)"
                       if m >= target and not ok else ""))
                record_found(found, m, pts)
                if best is None or (ok, m) > (best_ok, best[0]):
                    best, best_ok = (m, pts, f"subset:{name}"), ok
                if ok:
                    Path(stop_path).touch()
        if best_ok:
            break
    return best


# ---------------------------------------------------------------------------
# stage (b): diverse beam over exact coords
# ---------------------------------------------------------------------------

def _complete_to_n(P, n, deadline):
    """Greedy add-best until len(P) == n (min-sep-safe); None if stuck."""
    while len(P) < n:
        if time.time() > deadline + 60:
            return None
        U = universe(P, 1)
        U = U[~sep_viol_mask(U, P).any(axis=1)]
        if len(U) == 0:
            return None
        g = c40_unit_mask(U, P).sum(axis=1)
        P = np.vstack([P, U[int(np.argmax(g))][None, :]])
    return P


def _beam_task(payload):
    (n, width, seed, minutes, stop_path, pair_threshold, target) = payload
    rng = np.random.default_rng(seed)
    deadline = time.time() + minutes * 60
    stop = (lambda: Path(stop_path).exists()) if stop_path else (lambda: False)
    layers = beam_run(rng, width, n, deadline=deadline, should_stop=stop)
    best_c, best_P, best_ok = -1, None, False
    for lv in sorted(layers, reverse=True):
        members = layers[lv]
        take = 20 if lv >= n - 1 else 3
        for e, P in members[:take]:
            if stop() or time.time() > deadline + 60:
                break
            P = _complete_to_n(P, n, deadline)
            if P is None:
                continue
            P2, c = full_polish(P, pair_threshold=pair_threshold)
            ok = c >= target and audit_passes(P2)
            if (ok, c) > (best_ok, best_c):
                best_c, best_P, best_ok = c, P2, ok
            if best_ok:
                break
        if best_ok or stop():
            break
    pts = (
        [tuple(int(x) for x in r) for r in best_P.tolist()]
        if best_P is not None
        else None
    )
    return width, seed, best_c, pts, best_ok


def run_beam_stage(n, target, budget_s, processes, seed, stop_path, found):
    deadline = time.time() + budget_s
    best = None  # (edges, pts, label); audit-passing target hits outrank raw counts
    best_ok = False
    widths = [300, 700, 1500]
    for w, width in enumerate(widths):
        remaining = deadline - time.time()
        if remaining < 10:
            break
        slice_min = (remaining / (len(widths) - w)) / 60.0
        payloads = [
            (n, width, seed + 7700 + w * 97 + i, slice_min, stop_path,
             max(3, target - 2), target)
            for i in range(processes)
        ]
        with ProcessPoolExecutor(max_workers=processes) as ex:
            futs = [ex.submit(_beam_task, p) for p in payloads]
            for fut in as_completed(futs):
                wd, sd, c, pts, ok = fut.result()
                log(f"beam[w={wd} seed={sd}]: {c} edges"
                    + (" (target hit FAILS audit — not stopping)"
                       if pts is not None and c >= target and not ok else ""))
                if pts is not None:
                    record_found(found, c, pts)
                    if best is None or (ok, c) > (best_ok, best[0]):
                        best, best_ok = (c, pts, f"beam:w{wd}s{sd}"), ok
                    if ok:
                        Path(stop_path).touch()
        if best_ok:
            break
    return best


# ---------------------------------------------------------------------------
# stage (c): exact annealer
# ---------------------------------------------------------------------------

def run_anneal_stage(n, target, budget_s, processes, seed, out, found):
    """anneal_pool's own target stop is on the RAW count, and its pool is
    seeded from `found` (which may hold an audit-failing raw tie — then it
    returns instantly with zero generations run). So: first attempt with the
    target enabled (cheap legit early stop), and if the hit fails the float
    audit, re-loop once with target=None to actually anneal the remaining
    budget instead of re-stopping on the same bad config."""
    t_end = time.time() + budget_s
    best = None  # (edges, pts, label); audit-passing target hits outrank raw counts
    best_ok = False
    for attempt in (0, 1):
        starts = [pts for _e, pts in sorted(found.values(), key=lambda t: -t[0])[:10]]
        if not starts:
            starts = [greedy_add(list(tri_patch(1).points), k=n - 3)]
        best_e, best_pts, hist = anneal_pool(
            starts,
            minutes=max(0.05, (t_end - time.time()) / 60.0),
            procs=processes,
            steps=min(80_000, max(20_000, 800 * n)),
            t0=1.0,
            t1=0.05,
            reheats=2,
            seed=seed + 31337 + attempt * 7919,
            out=str(Path(out) / "anneal"),
            target=target if attempt == 0 else None,
            log=lambda m: log(f"anneal: {m}"),
        )
        pts = [tuple(p) for p in best_pts]
        record_found(found, best_e, pts)
        ok = best_e >= target and audit_passes(pts)
        if best is None or (ok, best_e) > (best_ok, best[0]):
            best, best_ok = (best_e, pts, "anneal"), ok
        if ok or best_e < target or time.time() >= t_end - 5:
            break
        log(f"anneal: raw target hit {best_e} FAILS audit — "
            "re-looping without the raw-count stop")
    return best


# ---------------------------------------------------------------------------
# stage (d): float-side polish (exploit_loop equivalent, in-process)
# ---------------------------------------------------------------------------

def run_polish_stage(best_pts, n, target, budget_s, processes, seed):
    """Warm Metropolis + coincidence forcing, monotone, audited every step.
    Returns (count, P_float, label) -- possibly off-lattice if forcing fired."""
    deadline = time.time() + budget_s
    P = to_float(MLConfig(best_pts))
    rep = audit(P)
    if not rep.passed:
        log("polish: exact best fails float audit?! skipping polish")
        return None
    count = rep.n_edges
    rnd = 0
    while time.time() < deadline - 10 and count < target:
        rnd += 1
        gained = False
        steps = 60_000
        results = multi_search(
            len(P),
            list(range(seed + 900 + rnd * 16, seed + 900 + rnd * 16 + processes)),
            steps,
            processes=processes,
            P0=P,
            T0=0.2,
            T1=0.01,
        )
        cand = None
        for r in results:
            a = audit(r.P)
            if a.passed and a.n_edges > count and (cand is None or a.n_edges > cand[1]):
                cand = (r.P, a.n_edges)
        if cand is not None:
            P, count = cand
            gained = True
            log(f"polish round {rnd}: search -> {count}")
        if time.time() < deadline:
            Q, _edges, _trail = force(P, log=lambda m: None)
            a = audit(Q)
            if a.passed and a.n_edges > count:
                P, count = Q, a.n_edges
                gained = True
                log(f"polish round {rnd}: forcing -> {count}")
        if not gained:
            log(f"polish round {rnd}: dry — stopping")
            break
    return (count, P, "polish")


# ---------------------------------------------------------------------------
# bookkeeping: found pool, certification, saving
# ---------------------------------------------------------------------------

def record_found(found: dict, e: int, pts) -> None:
    """Keep the best distinct classes seen (canonical key, capped at 24)."""
    if pts is None:
        return
    key = canon(np.array(pts, dtype=np.int64))
    if key not in found or found[key][0] < e:
        found[key] = (e, [tuple(int(x) for x in p) for p in pts])
    if len(found) > 24:
        for k in sorted(found, key=lambda k: found[k][0])[: len(found) - 24]:
            del found[k]


def certify_and_save(out: Path, n: int, stage: str, pts=None, P_float=None):
    """Float three-audit + exact certification + save CSV/coords/audit/cert.

    Either pts (exact integer 4-tuples) or P_float ((n,2) float) given.
    Returns a summary dict.
    """
    out.mkdir(parents=True, exist_ok=True)
    if P_float is None:
        P_float = to_float(MLConfig(pts))
    rep = audit(P_float)
    stem = f"udg{n}_{rep.n_edges}edges_{stage}"
    csv_path = out / f"{stem}.csv"
    save_csv(P_float, csv_path)
    save_audit_json(rep, out / f"{stem}_audit.json")
    cert = certify_config(load_csv(csv_path), name=str(csv_path))
    with open(out / f"{stem}_cert.json", "w") as f:
        json.dump(cert, f, indent=1)
    coords = pts
    if coords is None and cert.get("certified"):
        coords = [tuple(c) for c in cert["coords"]]
    if coords is not None:
        with open(out / f"{stem}_coords.json", "w") as f:
            json.dump(
                {"n": n, "exact_edges": edge_count(np.array(coords, dtype=np.int64)),
                 "coords": [list(p) for p in coords]},
                f,
            )
    info = {
        "stage": stage,
        "edges": rep.n_edges,
        "audit_passed": bool(rep.passed),
        "certified": bool(cert.get("certified", False)),
        "min_sep": rep.min_sep,
        "k23_violations": rep.k23_violations,
        "gn_total_residual": rep.gn_total_residual,
        "files": {
            "csv": str(csv_path),
            "audit": str(out / f"{stem}_audit.json"),
            "cert": str(out / f"{stem}_cert.json"),
            "coords": str(out / f"{stem}_coords.json") if coords is not None else None,
        },
    }
    if not rep.passed:
        log(f"WARNING: {stem} FAILED the float three-audit")
    if not cert.get("certified", False):
        log(f"note: {stem} not exactly certified (off-lattice or degenerate)")
    return info, coords


# ---------------------------------------------------------------------------
# Engel DB novelty diff
# ---------------------------------------------------------------------------

def engel_db_diff(pts, n: int, our_count: int) -> dict:
    """Scan the full local Engel DB at this n: exact batch edge counts
    (engel_targets recipe) + lattice identity via udg.mlgraph.canon."""
    if not NPZ.exists():
        return {"available": False, "note": "runs/engel_db/seen_graphs.npz missing"}
    z = np.load(NPZ, allow_pickle=False)
    keys = sorted(
        (k for k in z.files if k.endswith(f"_{n}")), key=lambda k: int(k.split("_")[0])
    )
    our_key = canon(np.array(pts, dtype=np.int64))
    their_max = -1
    n_total = 0
    n_at_or_above = 0
    lattice_identical = False
    chunk = max(64, (1 << 24) // max(n * n * 4, 1))
    for k in keys:
        arr = np.asarray(z[k], dtype=np.int64)
        n_total += len(arr)
        for i0 in range(0, len(arr), chunk):
            X = arr[i0 : i0 + chunk]
            ec = batch_edge_counts(X)
            m = int(ec.max()) if len(ec) else -1
            their_max = max(their_max, m)
            for h in np.nonzero(ec >= our_count)[0]:
                n_at_or_above += 1
                if canon(X[int(h)]) == our_key:
                    lattice_identical = True
    return {
        "available": True,
        "n_db_configs": n_total,
        "their_max_edges": their_max,
        "db_configs_at_our_count_or_above": n_at_or_above,
        "lattice_identical_in_db": lattice_identical,
        "novel": (our_count > their_max) and not lattice_identical,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--target", type=int, default=None,
                    help="record to chase (default: runs/engel_db/targets.json)")
    ap.add_argument("--budget-min", type=float, default=12.0,
                    help="total wall-clock budget in minutes")
    ap.add_argument("--out", default=None, help="default runs/sweep/n<n>/")
    ap.add_argument("--processes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stages", default="subset,beam,anneal,polish",
                    help="comma-separated subset of the ladder to run")
    args = ap.parse_args()

    n = args.n
    target, target_src = resolve_target(n, args.target)
    out = Path(args.out) if args.out else ROOT / f"runs/sweep/n{n}"
    out.mkdir(parents=True, exist_ok=True)
    stop_path = str(out / ".stop")
    Path(stop_path).unlink(missing_ok=True)
    sanity_check_arithmetic()

    log(f"n={n} target={target} ({target_src}) budget={args.budget_min}min "
        f"processes={args.processes} seed={args.seed} out={out}")

    t_start = time.time()
    budget_s = args.budget_min * 60.0
    found: dict = {}  # canon -> (edges, pts)
    summary: dict = {
        "n": n, "target": target, "target_source": target_src,
        "seed": args.seed, "budget_min": args.budget_min,
        "processes": args.processes, "stages": [],
    }
    global_best = None  # (edges, pts_or_None, label, P_float_or_None, audit_ok)

    def remaining() -> float:
        return budget_s - (time.time() - t_start)

    stages = [s.strip() for s in args.stages.split(",") if s.strip() in STAGE_WEIGHTS]
    if not stages:
        raise SystemExit(f"--stages must name some of {list(STAGE_WEIGHTS)}")
    summary["stages_requested"] = stages
    for si, stage in enumerate(stages):
        if ladder_done(global_best, target):
            break
        rem_weights = sum(STAGE_WEIGHTS[s] for s in stages[si:])
        slice_s = max(0.0, remaining()) * STAGE_WEIGHTS[stage] / rem_weights
        if slice_s < 10:
            summary["stages"].append({"name": stage, "skipped": "budget exhausted"})
            continue
        t0 = time.time()
        log(f"=== stage {stage}: {slice_s/60:.1f} min ===")
        result = None
        if stage == "subset":
            result = run_subset_stage(
                n, target, slice_s, args.processes, args.seed, stop_path, found
            )
        elif stage == "beam":
            result = run_beam_stage(
                n, target, slice_s, args.processes, args.seed, stop_path, found
            )
        elif stage == "anneal":
            result = run_anneal_stage(
                n, target, slice_s, args.processes, args.seed, out, found
            )
        elif stage == "polish":
            if global_best is None:
                summary["stages"].append({"name": stage, "skipped": "nothing to polish"})
                continue
            result = run_polish_stage(
                global_best[1], n, target, slice_s, args.processes, args.seed
            )
        secs = time.time() - t0
        Path(stop_path).unlink(missing_ok=True)

        entry = {"name": stage, "seconds": round(secs, 1)}
        if result is None:
            entry["best"] = None
            summary["stages"].append(entry)
            continue
        if stage == "polish":
            count, P_float, label = result
            ok = True  # polish only ever returns audit-passing counts
            improved = (global_best is None
                        or (ok, count) > (global_best[4], global_best[0]))
            entry["best"] = count
            if improved:
                info, coords = certify_and_save(out, n, label, P_float=P_float)
                entry["claim"] = info
                global_best = (count, coords, label, P_float, ok)
        else:
            m, pts, label = result
            ok = audit_passes(pts)
            improved = (global_best is None
                        or (ok, m) > (global_best[4], global_best[0]))
            entry["best"] = m
            entry["winner"] = label
            if improved:
                info, _ = certify_and_save(out, n, label.replace(":", "_"), pts=pts)
                entry["claim"] = info
                global_best = (m, pts, label, None, ok)
        entry["audit_passed"] = ok
        entry["hit_target"] = bool(entry.get("best") is not None
                                   and entry["best"] >= target and ok)
        summary["stages"].append(entry)
        log(f"stage {stage}: best {entry['best']} in {secs:.1f}s"
            + (" (TARGET HIT)" if entry["hit_target"] else "")
            + ("" if ok else " (FAILS AUDIT — not a claim)"))

    # ----- final claim -----------------------------------------------------
    if global_best is None:
        log("no configuration found (?)")
        summary["best"] = None
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        return 1

    best_edges, best_pts, best_label, best_float, _best_ok = global_best
    if best_pts is not None:
        final_info, final_coords = certify_and_save(
            out, n, "final", pts=best_pts
        )
    else:
        final_info, final_coords = certify_and_save(
            out, n, "final", P_float=best_float
        )
    summary["best"] = {
        **final_info,
        "edges": best_edges,
        "stage": best_label,
        "target": target,
        **final_verdict(best_edges, target, final_info),
    }

    if best_edges > target and final_info["audit_passed"]:
        print("=" * 72, flush=True)
        print(f"*** NEW RECORD CANDIDATE: n={n} edges={best_edges} > "
              f"target={target} (stage {best_label}) ***", flush=True)
        print("=" * 72, flush=True)
        if final_coords is not None:
            diff = engel_db_diff(final_coords, n, best_edges)
            summary["engel_diff"] = diff
            log(f"Engel DB diff: {diff}")
            if diff.get("novel"):
                print(f"*** NOT IN THE ENGEL DB at >= {best_edges} edges: "
                      "this is the real thing — run the Tier-3 protocol ***",
                      flush=True)
        else:
            summary["engel_diff"] = {
                "available": False,
                "note": "no exact coords (off-lattice candidate) — "
                        "diff requires manual iso check (scripts/engel_iso.py)",
            }

    elapsed = time.time() - t_start
    summary["elapsed_seconds"] = round(elapsed, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    log(f"DONE: best {best_edges} / target {target} in {elapsed:.0f}s"
        + ("" if final_info["audit_passed"] else " (FAILS AUDIT — no claim)")
        + f" -> {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
