"""Exact graph-isomorphism pass over runs/engel_db/classes_n{n}.json.

Run with:  uv run --with networkx python scripts/engel_iso.py

For each n: groups our lattice-class reps into exact iso classes (VF2),
groups Engel's lattice-class reps likewise, then matches our iso classes
against theirs. Verdict per our class: IDENTICAL (lattice), ISO (same graph,
different embedding), or NEW (no isomorphic Engel record graph).
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
RECORDS = {30: 93, 40: 137, 50: 183, 70: 281}


def to_graph(n: int, edges: list[list[int]]) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(map(tuple, edges))
    return G


def iso_group(graphs: list[nx.Graph]) -> list[int]:
    """Assign each graph an iso-class id (pairwise VF2 within the list)."""
    ids: list[int] = []
    reps: list[nx.Graph] = []
    for G in graphs:
        for cid, R in enumerate(reps):
            if nx.is_isomorphic(G, R):
                ids.append(cid)
                break
        else:
            reps.append(G)
            ids.append(len(reps) - 1)
    return ids


def main() -> None:
    for n, want in RECORDS.items():
        p = ROOT / f"runs/engel_db/classes_n{n}.json"
        if not p.exists():
            print(f"n={n}: {p.name} missing — run engel_compare.py compare first")
            continue
        d = json.loads(p.read_text())
        ours, theirs = d["ours"], d["theirs"]
        Go = [to_graph(n, o["edges"]) for o in ours]
        Gt = [to_graph(n, t["edges"]) for t in theirs]
        oid = iso_group(Go)
        tid_offset = len(set(oid))
        # group theirs against OUR reps first so shared classes get our ids
        tids: list[int] = []
        our_reps: dict[int, nx.Graph] = {}
        for i, G in enumerate(Go):
            our_reps.setdefault(oid[i], G)
        extra: list[nx.Graph] = []
        for G in Gt:
            for cid, R in our_reps.items():
                if nx.is_isomorphic(G, R):
                    tids.append(cid)
                    break
            else:
                for j, R in enumerate(extra):
                    if nx.is_isomorphic(G, R):
                        tids.append(tid_offset + j)
                        break
                else:
                    extra.append(G)
                    tids.append(tid_offset + len(extra) - 1)
        n_ours = len(set(oid))
        n_theirs = len(set(tids))
        shared = len(set(oid) & set(tids))
        print(f"\nn={n} ({want} edges): our iso classes {n_ours}, "
              f"Engel iso classes {n_theirs}, shared {shared}, "
              f"ours-NEW {n_ours - shared}, theirs-not-found-by-us {n_theirs - shared}")
        for i, o in enumerate(ours):
            cid = oid[i]
            if o["lattice_identical"]:
                verdict = "IDENTICAL (lattice-congruent to an Engel config)"
            elif cid in set(tids):
                verdict = "ISO (isomorphic to an Engel record graph, different embedding)"
            else:
                verdict = "NEW (no isomorphic graph among Engel record configs)"
            print(f"  ours[{i}] iso-class {cid}  {o['rep']}: {verdict}")
        for j, t in enumerate(theirs):
            tag = "shared" if tids[j] in set(oid) else "THEIRS-ONLY"
            print(f"  engel[{j}] idx {t['idx']} iso-class {tids[j]} ({tag})")


if __name__ == "__main__":
    main()
