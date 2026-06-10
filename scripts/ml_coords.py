#!/usr/bin/env python
"""Exact Moser-lattice coordinate recovery + certification for UDG configs.

The Moser lattice ML = Z<1, w1, w3, w1*w3> (w1 = exp(i*pi/3), w3 =
exp(i*arccos(5/6))) is a rank-4 free Z-module densely embedded in C, so
integer coordinates CANNOT be recovered by rounding float positions; they
must be propagated combinatorially along unit edges. This script:

1. loads a config CSV, takes its unit edges (tol 1e-9);
2. refines the global rotation: seed from udg.moser.lattice_id, then
   alternate (match each edge vector to the nearest of the 18 ML unit
   vectors) <-> (circular-mean rotation update) until the assignment is
   stable; reports the max angular residual (<= ~1e-5 rad for genuine ML);
3. anchors a BFS spanning-tree root at (0,0,0,0) and propagates integer
   coordinates child = parent +/- coeffs(matched unit vector);
4. cycle check: every non-tree edge's integer-coordinate difference must
   equal the matched unit vector's coefficient tuple EXACTLY;
5. verifies EXACTLY, no floats: arithmetic in Q(sqrt3, sqrt11) with basis
   (1, sqrt3, sqrt11, sqrt33) as 4-tuples of fractions.Fraction; for every
   point pair |p_i - p_j|^2 is computed exactly and counted iff == 1; all
   pairs must additionally be distinct (|diff|^2 != 0);
6. floats-vs-exact consistency: rigid Procrustes alignment of the float
   positions onto the exact embedding; max per-point error must be tiny.
   (Without this gate a config jittered by ~1e-3 would still "certify":
   the nearest-unit-vector matching is robust to ~8 deg of angular noise,
   so the propagated integer coords and the cycle check are unchanged by
   small jitter -- the combinatorial certificate would be valid for the
   IDEAL config but say nothing about the floats on disk.)

certified = single component AND cycle_failures == 0 AND distinct_ok
            AND exact_unit_pairs >= float edge count
            AND max angular residual < 1e-4 rad
            AND Procrustes max embedding error < 1e-6.

If exact_unit_pairs EXCEEDS the float edge count, the exact embedding has
unit pairs the float tolerance missed (bonus edges) -- reported prominently.

Output: runs/mlcoords/<name>.json per config + a one-line verdict.
Exit code = number of configs that failed certification.

Usage:
    uv run python scripts/ml_coords.py data/udg40_136edges.csv [more.csv ...]
        [--outdir runs/mlcoords] [--tol 1e-9]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

from udg.configio import load_csv
from udg.counting import unit_edges
from udg.moser import UNIT_COEFFS, W1, W3, lattice_id, ml_directions, unit_vectors

TOL = 1e-9              # unit-distance tolerance for float configs
ANG_RESID_GATE = 1e-4   # rad; genuine ML configs come in at <= ~1e-5
EMBED_ERR_GATE = 1e-6   # max |float - aligned exact| per point; genuine ~1e-12

_SQRT3 = math.sqrt(3.0)
_SQRT11 = math.sqrt(11.0)
_SQRT33 = math.sqrt(33.0)


# ---------------------------------------------------------------------------
# Exact arithmetic in Q(sqrt3, sqrt11), basis (1, sqrt3, sqrt11, sqrt33)
# ---------------------------------------------------------------------------

class QF:
    """Element a + b*sqrt3 + c*sqrt11 + d*sqrt33 of Q(sqrt3, sqrt11).

    Coordinates are fractions.Fraction; all operations are exact. The
    multiplication table of the basis (1, s3, s11, s33):

        s3*s3 = 3        s11*s11 = 11      s33*s33 = 33
        s3*s11 = s33     s3*s33 = 3*s11    s11*s33 = 11*s3
    """

    __slots__ = ("a", "b", "c", "d")

    def __init__(self, a=0, b=0, c=0, d=0):
        self.a = Fraction(a)
        self.b = Fraction(b)
        self.c = Fraction(c)
        self.d = Fraction(d)

    def __add__(self, o: "QF") -> "QF":
        return QF(self.a + o.a, self.b + o.b, self.c + o.c, self.d + o.d)

    def __sub__(self, o: "QF") -> "QF":
        return QF(self.a - o.a, self.b - o.b, self.c - o.c, self.d - o.d)

    def __mul__(self, o: "QF") -> "QF":
        a1, b1, c1, d1 = self.a, self.b, self.c, self.d
        a2, b2, c2, d2 = o.a, o.b, o.c, o.d
        return QF(
            a1 * a2 + 3 * b1 * b2 + 11 * c1 * c2 + 33 * d1 * d2,
            a1 * b2 + b1 * a2 + 11 * (c1 * d2 + d1 * c2),
            a1 * c2 + c1 * a2 + 3 * (b1 * d2 + d1 * b2),
            a1 * d2 + d1 * a2 + b1 * c2 + c1 * b2,
        )

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, QF):
            return NotImplemented
        return self.a == o.a and self.b == o.b and self.c == o.c and self.d == o.d

    def __hash__(self) -> int:
        return hash((self.a, self.b, self.c, self.d))

    def is_zero(self) -> bool:
        return self.a == 0 and self.b == 0 and self.c == 0 and self.d == 0

    def to_float(self) -> float:
        return (
            float(self.a)
            + float(self.b) * _SQRT3
            + float(self.c) * _SQRT11
            + float(self.d) * _SQRT33
        )

    def __repr__(self) -> str:
        return f"QF({self.a}, {self.b}, {self.c}, {self.d})"


QF_ZERO = QF()
QF_ONE = QF(1)

# Exact real/imag parts of the ML basis (1, w1, w3, w1*w3) in Q(sqrt3, sqrt11):
#   w1   = 1/2 + (sqrt3/2) i
#   w3   = 5/6 + (sqrt11/6) i
#   w1w3 = 5/12 - sqrt33/12 + (sqrt11/12 + 5*sqrt3/12) i
_RE_BASIS = (
    QF(1),
    QF(Fraction(1, 2)),
    QF(Fraction(5, 6)),
    QF(Fraction(5, 12), 0, 0, Fraction(-1, 12)),
)
_IM_BASIS = (
    QF(0),
    QF(0, Fraction(1, 2)),
    QF(0, 0, Fraction(1, 6)),
    QF(0, Fraction(5, 12), Fraction(1, 12), 0),
)


def exact_xy(coeffs) -> tuple[QF, QF]:
    """Exact (Re, Im) in Q(sqrt3, sqrt11) of an integer ML coordinate 4-tuple."""
    re = QF_ZERO
    im = QF_ZERO
    for m, rb, ib in zip(coeffs, _RE_BASIS, _IM_BASIS):
        if m:
            mq = QF(m)
            re = re + mq * rb
            im = im + mq * ib
    return re, im


def exact_dist2(ca, cb) -> QF:
    """Exact |p(ca) - p(cb)|^2 for two integer ML coordinate 4-tuples."""
    ra, ia = exact_xy(ca)
    rb, ib = exact_xy(cb)
    dre = ra - rb
    dim = ia - ib
    d2 = dre * dre + dim * dim
    # structural invariant: Re, Im of ML points live in Q+Q*s33 and
    # Q*s3+Q*s11 respectively, so |.|^2 has no s3/s11 component
    assert d2.b == 0 and d2.c == 0, f"impossible dist^2 {d2!r}"
    return d2


def sanity_check_arithmetic() -> None:
    """Validate the exact field arithmetic against the 18 ML unit vectors.

    1. every UNIT_COEFFS tuple has |v|^2 == 1 EXACTLY;
    2. exact Re/Im reproduce the float unit vectors to 1e-12;
    3. pairwise sums and differences of unit vectors reproduce the float
       values to 1e-12 (exercises add/sub/mul on nontrivial elements).
    """
    uv = unit_vectors()
    origin = (0, 0, 0, 0)
    for k, c in enumerate(UNIT_COEFFS):
        d2 = exact_dist2(c, origin)
        if d2 != QF_ONE:
            raise AssertionError(f"unit coeff {c} has |v|^2 = {d2!r} != 1")
        re, im = exact_xy(c)
        if abs(re.to_float() - uv[k].real) > 1e-12 or abs(im.to_float() - uv[k].imag) > 1e-12:
            raise AssertionError(f"unit coeff {c} float mismatch")
    for i in range(len(UNIT_COEFFS)):
        for j in range(len(UNIT_COEFFS)):
            s = tuple(a + b for a, b in zip(UNIT_COEFFS[i], UNIT_COEFFS[j]))
            re, im = exact_xy(s)
            z = uv[i] + uv[j]
            if abs(re.to_float() - z.real) > 1e-12 or abs(im.to_float() - z.imag) > 1e-12:
                raise AssertionError(f"sum uv[{i}]+uv[{j}] float mismatch")
            d = tuple(a - b for a, b in zip(UNIT_COEFFS[i], UNIT_COEFFS[j]))
            re, im = exact_xy(d)
            z = uv[i] - uv[j]
            if abs(re.to_float() - z.real) > 1e-12 or abs(im.to_float() - z.imag) > 1e-12:
                raise AssertionError(f"diff uv[{i}]-uv[{j}] float mismatch")


# ---------------------------------------------------------------------------
# Rotation refinement (step 2)
# ---------------------------------------------------------------------------

def refine_rotation(
    P: np.ndarray, edges, theta0: float, max_iter: int = 64
) -> tuple[float, np.ndarray, float]:
    """Alternate nearest-unit-vector matching <-> circular-mean rotation.

    theta is the angle (radians) such that edge vectors v ~= e^{i theta} * u
    for ML unit vectors u; equivalently v * e^{-i theta} matches the 18
    unit vectors (sign handled automatically: the 18 come in +/- pairs).

    Returns (theta, assign, max_angular_residual_rad) where assign[k] is
    the index into UNIT_COEFFS matched to edge k (vector P[j] - P[i]).
    """
    E = np.asarray(list(edges), dtype=int).reshape(-1, 2)
    v = (P[E[:, 1], 0] - P[E[:, 0], 0]) + 1j * (P[E[:, 1], 1] - P[E[:, 0], 1])
    vhat = v / np.abs(v)
    uv = unit_vectors()
    theta = float(theta0)
    assign = None
    for _ in range(max_iter):
        rotated = vhat * np.exp(-1j * theta)
        new_assign = np.argmin(np.abs(rotated[:, None] - uv[None, :]), axis=1)
        # circular-mean update: phasors vhat * conj(uv[assign]) ~ e^{i theta}
        ph = vhat * np.conj(uv[new_assign])
        new_theta = float(np.angle(ph.sum()))
        if (
            assign is not None
            and np.array_equal(new_assign, assign)
            and abs(new_theta - theta) < 1e-15
        ):
            assign, theta = new_assign, new_theta
            break
        assign, theta = new_assign, new_theta
    resid = np.angle(vhat * np.exp(-1j * theta) * np.conj(uv[assign]))
    return theta, assign, float(np.max(np.abs(resid)))


# ---------------------------------------------------------------------------
# Spanning tree propagation + cycle check (steps 3-4)
# ---------------------------------------------------------------------------

def propagate_coords(
    n: int, edges: list[tuple[int, int]], assign: np.ndarray
) -> tuple[list[tuple[int, int, int, int]], list[list[int]], list[dict]]:
    """BFS spanning tree per component, anchor roots at (0,0,0,0), propagate
    child = parent +/- UNIT_COEFFS[assign[k]], then check every non-tree edge.

    Returns (coords, components, cycle_failures). cycle_failures entries:
    {edge: [i, j], expected: coeffs, observed: coords[j]-coords[i]}.
    """
    adj: list[list[tuple[int, int, int]]] = [[] for _ in range(n)]
    for k, (i, j) in enumerate(edges):
        adj[i].append((j, k, +1))   # traversing i->j adds +coeffs
        adj[j].append((i, k, -1))
    coords: list[tuple[int, int, int, int] | None] = [None] * n
    components: list[list[int]] = []
    tree_edges: set[int] = set()
    for root in range(n):
        if coords[root] is not None:
            continue
        coords[root] = (0, 0, 0, 0)
        comp = [root]
        queue = [root]
        while queue:
            u = queue.pop(0)
            for v, k, s in adj[u]:
                if coords[v] is None:
                    c = UNIT_COEFFS[assign[k]]
                    cu = coords[u]
                    coords[v] = (
                        cu[0] + s * c[0],
                        cu[1] + s * c[1],
                        cu[2] + s * c[2],
                        cu[3] + s * c[3],
                    )
                    tree_edges.add(k)
                    comp.append(v)
                    queue.append(v)
        components.append(comp)

    cycle_failures: list[dict] = []
    for k, (i, j) in enumerate(edges):
        if k in tree_edges:
            continue  # holds by construction
        expected = UNIT_COEFFS[assign[k]]
        observed = tuple(a - b for a, b in zip(coords[j], coords[i]))
        if observed != expected:
            cycle_failures.append(
                {"edge": [i, j], "expected": list(expected), "observed": list(observed)}
            )
    return coords, components, cycle_failures  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Certification (steps 5-6)
# ---------------------------------------------------------------------------

def certify_config(P: np.ndarray, name: str = "", tol: float = TOL) -> dict:
    """Run the full ML-coordinate certification on a float config."""
    P = np.asarray(P, dtype=float)
    n = len(P)
    edges = unit_edges(P, tol=tol)
    n_float = len(edges)
    result: dict = {
        "config": name,
        "n": n,
        "float_unit_edges": n_float,
        "certified": False,
    }
    if n_float == 0:
        result["notes"] = "no unit edges at tol -- nothing to certify"
        return result

    # step 2: rotation
    lid = lattice_id(P, edges)
    theta0 = math.radians(lid["best_rotation"])
    theta, assign, max_resid = refine_rotation(P, edges, theta0)
    result["rotation_deg"] = math.degrees(theta) % 180.0
    result["max_angular_residual_rad"] = max_resid
    result["lattice_id_matched"] = f"{lid['n_matched']}/{lid['n_dirs']}"

    # per-direction-family characterization (which families fail to match)
    mld = ml_directions()
    fam = []
    E = np.asarray(edges, dtype=int)
    vec = (P[E[:, 1], 0] - P[E[:, 0], 0]) + 1j * (P[E[:, 1], 1] - P[E[:, 0], 1])
    resid_all = np.abs(
        np.angle(vec / np.abs(vec) * np.exp(-1j * theta) * np.conj(unit_vectors()[assign]))
    )
    for u_idx in sorted(set(int(a) for a in assign)):
        mask = assign == u_idx
        ang = math.degrees(np.angle(unit_vectors()[u_idx])) % 180.0
        fam.append(
            {
                "ml_dir_deg": round(min(mld, key=lambda m: min(abs(ang - m), 180 - abs(ang - m))), 3),
                "unit_vector": list(UNIT_COEFFS[u_idx]),
                "edges": int(mask.sum()),
                "max_residual_rad": float(resid_all[mask].max()),
            }
        )
    result["families"] = fam

    # steps 3-4: spanning tree propagation + cycle check
    coords, components, cycle_failures = propagate_coords(n, edges, assign)
    result["n_components"] = len(components)
    result["coords"] = [list(c) for c in coords]
    result["cycle_failures"] = len(cycle_failures)
    result["cycle_failure_details"] = cycle_failures[:50]
    if len(components) > 1:
        result["notes"] = (
            f"edge graph is DISCONNECTED ({len(components)} components incl. "
            "isolated vertices) -- coords anchored per component, cross-component "
            "exact distances are meaningless; cannot certify"
        )
        return result

    # step 5: exact verification over all pairs, no floats
    exact_unit_pairs = 0
    distinct_ok = True
    unit_pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            d2 = exact_dist2(coords[i], coords[j])
            if d2 == QF_ONE:
                exact_unit_pairs += 1
                unit_pairs.append((i, j))
            elif d2.is_zero():
                distinct_ok = False
    result["exact_unit_pairs"] = exact_unit_pairs
    result["distinct_ok"] = distinct_ok

    edge_set = {(min(i, j), max(i, j)) for i, j in edges}
    exact_set = set(unit_pairs)
    bonus = sorted(exact_set - edge_set)
    missing = sorted(edge_set - exact_set)
    result["bonus_pairs"] = [list(p) for p in bonus]
    result["missing_pairs"] = [list(p) for p in missing]

    # step 6 extra gate: floats are (rigid motion of) the exact embedding
    z = np.array(
        [complex(exact_xy(c)[0].to_float(), exact_xy(c)[1].to_float()) for c in coords]
    )
    p = P[:, 0] + 1j * P[:, 1]
    zc = z - z.mean()
    pc = p - p.mean()
    s = (pc * np.conj(zc)).sum()
    rot = s / abs(s) if abs(s) > 0 else 1.0
    embed_err = float(np.max(np.abs(pc - rot * zc)))
    result["embed_max_err"] = embed_err

    result["certified"] = (
        len(cycle_failures) == 0
        and distinct_ok
        and exact_unit_pairs >= n_float
        and max_resid < ANG_RESID_GATE
        and embed_err < EMBED_ERR_GATE
    )
    return result


def _verdict_line(r: dict) -> str:
    name = Path(r["config"]).name if r["config"] else "<config>"
    if r["certified"]:
        line = (
            f"{name}: CERTIFIED ML  n={r['n']} float_edges={r['float_unit_edges']} "
            f"exact_unit_pairs={r['exact_unit_pairs']} rot={r['rotation_deg']:.6f} deg "
            f"max_resid={r['max_angular_residual_rad']:.2e} rad "
            f"embed_err={r['embed_max_err']:.2e}"
        )
        if r["exact_unit_pairs"] > r["float_unit_edges"]:
            line += (
                f"  *** BONUS: {r['exact_unit_pairs'] - r['float_unit_edges']} exact "
                f"unit pair(s) MISSED by float tol: {r['bonus_pairs']} ***"
            )
        return line
    reasons = []
    if r.get("n_components", 1) > 1:
        reasons.append(f"disconnected ({r['n_components']} components)")
    if r.get("cycle_failures", 0):
        reasons.append(f"cycle_failures={r['cycle_failures']}")
    if not r.get("distinct_ok", True):
        reasons.append("coincident exact points")
    if r.get("exact_unit_pairs", 0) < r.get("float_unit_edges", 0):
        reasons.append(
            f"exact_unit_pairs={r.get('exact_unit_pairs')} < float_edges={r['float_unit_edges']}"
        )
    if r.get("max_angular_residual_rad", 0.0) >= ANG_RESID_GATE:
        reasons.append(f"max_resid={r['max_angular_residual_rad']:.2e} rad >= {ANG_RESID_GATE:.0e}")
    if r.get("embed_max_err", 0.0) >= EMBED_ERR_GATE:
        reasons.append(f"embed_err={r['embed_max_err']:.2e} >= {EMBED_ERR_GATE:.0e}")
    if "notes" in r:
        reasons.append(r["notes"])
    return f"{name}: NOT CERTIFIED ({'; '.join(reasons) or 'unknown'})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("csv", nargs="+", help="config CSV(s) with header x,y")
    ap.add_argument("--outdir", default="runs/mlcoords", help="JSON output directory")
    ap.add_argument("--tol", type=float, default=TOL, help="unit-distance tolerance")
    args = ap.parse_args(argv)

    sanity_check_arithmetic()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    n_failed = 0
    for path in args.csv:
        P = load_csv(path)
        r = certify_config(P, name=str(path), tol=args.tol)
        out = outdir / (Path(path).stem + ".json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2)
            f.write("\n")
        print(_verdict_line(r))
        if not r["certified"]:
            n_failed += 1
            # characterize HOW it fails: per-family residuals
            for fam in r.get("families", []):
                flag = " <-- OFF-LATTICE" if fam["max_residual_rad"] > ANG_RESID_GATE else ""
                print(
                    f"    family ml_dir={fam['ml_dir_deg']:7.3f} deg  "
                    f"edges={fam['edges']:3d}  max_resid={fam['max_residual_rad']:.3e} rad{flag}"
                )
            for cf in r.get("cycle_failure_details", [])[:10]:
                print(
                    f"    cycle FAIL edge {cf['edge']}: expected {cf['expected']}, "
                    f"observed {cf['observed']}"
                )
    return n_failed


if __name__ == "__main__":
    sys.exit(main())
