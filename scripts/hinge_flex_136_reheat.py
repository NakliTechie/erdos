"""EXP2d — hotter reheat from a 136-edge config chasing the n=40 record (137).

The 136 configs are rigid and ML-locked with no near-misses < 8.2e-2, so the
flex recipe is exhausted; only discrete restructuring can gain more. One batch:
4 seeds x 60k steps at T0=0.5 from flex_135A_search_seed401.csv.
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
    P0 = load_csv(OUT / "flex_135A_search_seed401.csv")
    res = multi_search(40, [421, 422, 423, 424], 60_000, processes=4,
                       P0=P0, T0=0.5, T1=0.01)
    rows = []
    for sr in res:
        rep = audit(sr.P)
        row = dict(seed=sr.seed, raw=int(sr.best_count), n_edges=int(rep.n_edges),
                   passed=bool(rep.passed), min_sep=float(rep.min_sep),
                   min_sep_after=float(rep.min_sep_after))
        if rep.passed and rep.n_edges >= 132 and not np.array_equal(sr.P, P0):
            stem = f"flex_136_reheat_seed{sr.seed}"
            save_csv(sr.P, OUT / f"{stem}.csv")
            save_audit_json(rep, OUT / f"{stem}_audit.json")
            row["saved"] = stem + ".csv"
        rows.append(row)
        print(row)
    json.dump(rows, open(OUT / "flex_136_reheat_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
