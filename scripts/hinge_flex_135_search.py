"""EXP2c — low-T warm-start searches from the two 135-edge crossing configs.

Both flex_cross 135 configs (s=-0.040 and s=+0.570 events, fully ML-locked,
flex_dim 0) get 4 seeds x 30k at T0=0.08; the s=-0.040 one also gets a
moderate-reheat batch (T0=0.35) to probe whether discrete restructuring can
exceed 135. Everything audited; >=132 audited results saved unless byte-equal
to the warm start.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from udg.audit import audit
from udg.configio import load_csv, save_audit_json, save_csv
from udg.search import multi_search

OUT = Path("/Users/chiragpatnaik/Code/erdos/runs/hinge")


def main() -> None:
    out = {}
    for tag, src, seeds, T0 in [
        ("135A", "flex_cross_p5_11.csv", [401, 402, 403, 404], 0.08),
        ("135B", "flex_cross_p5_20.csv", [401, 402, 403, 404], 0.08),
        ("135A_reheat", "flex_cross_p5_11.csv", [411, 412, 413, 414], 0.35),
    ]:
        P0 = load_csv(OUT / src)
        res = multi_search(40, seeds, 30_000, processes=4, P0=P0, T0=T0, T1=0.01)
        rows = []
        for sr in res:
            rep = audit(sr.P)
            row = dict(
                seed=sr.seed,
                raw=int(sr.best_count),
                n_edges=int(rep.n_edges),
                passed=bool(rep.passed),
                min_sep=float(rep.min_sep),
                min_sep_after=float(rep.min_sep_after),
            )
            if rep.passed and rep.n_edges >= 132:
                if not np.array_equal(sr.P, P0):
                    stem = f"flex_{tag}_search_seed{sr.seed}"
                    save_csv(sr.P, OUT / f"{stem}.csv")
                    save_audit_json(rep, OUT / f"{stem}_audit.json")
                    row["saved"] = stem + ".csv"
                else:
                    row["saved"] = "identical to warm start (not duplicated)"
            rows.append(row)
            print(tag, row)
        out[tag] = rows
    json.dump(out, open(OUT / "flex_135_search_summary.json", "w"), indent=2)
    print("done")


if __name__ == "__main__":
    main()
