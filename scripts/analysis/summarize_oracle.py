#!/usr/bin/env python3
"""Summarize oracle single-expert probe records.

Per (eval set, layer band) and per checkpoint:
  Oracle@k = Pr(e* in S)         e* = argmin_e L(e), S = router top-k set
  Oracle@1 = Pr(e* == top1)
  R_set    = min_{e in S} L(e) - L(e*)
  R_rank   = L(top1) - min_{e in S} L(e)

Paired on identical (eval, seq, t, l); paired bootstrap (over sequences) 95%
CIs on the baseline-vs-drmoet DIFFERENCE of each metric. 'OOD macro' averages
the shifted S2ORC domains (excludes dclm and the cs.CL control).
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

LAYER_BAND = {0: "early", 4: "middle", 7: "late"}
CONTROLS = {"dclm", "cs.CL"}


def load(path):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[(r["eval"], r["seq"], r["t"], r["l"])] = r
    return out


def metrics(r):
    L = np.asarray(r["losses"])
    S = r["S"]
    estar = int(L.argmin())
    inS = estar in S
    at1 = estar == r["top1"]
    r_set = float(L[S].min() - L[estar])
    r_rank = float(L[r["top1"]] - L[S].min())
    return inS, at1, r_set, r_rank


def agg(rows):
    m = np.array([metrics(r) for r in rows], dtype=float)
    return m[:, 0].mean(), m[:, 1].mean(), m[:, 2].mean(), m[:, 3].mean()


def paired_boot(keys, B, D, stat_idx, n=2000, seed=0):
    """Bootstrap over sequences of the (dro - base) difference of metric i."""
    by_seq = defaultdict(list)
    for k in keys:
        by_seq[(k[0], k[1])].append(k)
    seqs = list(by_seq)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        pick = rng.choice(len(seqs), size=len(seqs), replace=True)
        ks = [k for i in pick for k in by_seq[seqs[i]]]
        mb = np.array([metrics(B[k]) for k in ks], dtype=float)[:, stat_idx].mean()
        md = np.array([metrics(D[k]) for k in ks], dtype=float)[:, stat_idx].mean()
        diffs.append(md - mb)
    return np.percentile(diffs, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="analysis/oracle_probe")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    B = load(os.path.join(args.dir, "records_baseline.jsonl"))
    D = load(os.path.join(args.dir, "records_drmoet.jsonl"))
    keys = sorted(set(B) & set(D))
    print(f"records: baseline={len(B)} drmoet={len(D)} paired={len(keys)}\n")
    assert keys, "no paired records"

    evals = sorted({k[0] for k in keys})
    names = ["Oracle@k", "Oracle@1", "R_set", "R_rank"]

    def show(tag, ks):
        mb = agg([B[k] for k in ks])
        md = agg([D[k] for k in ks])
        line = f"{tag:22}{len(ks):>6}"
        for i, nm in enumerate(names):
            if i < 2:
                line += f" |{mb[i]*100:6.1f}%{md[i]*100:6.1f}%"
            else:
                line += f" |{mb[i]:7.4f}{md[i]:7.4f}"
        print(line)
        ci_k = paired_boot(ks, B, D, 0, n=args.boot)
        ci_1 = paired_boot(ks, B, D, 1, n=args.boot)
        ci_s = paired_boot(ks, B, D, 2, n=args.boot)
        ci_r = paired_boot(ks, B, D, 3, n=args.boot)
        print(f"{'':22}{'':>6}  d@k[{ci_k[0]*100:+.1f},{ci_k[1]*100:+.1f}]%"
              f" d@1[{ci_1[0]*100:+.1f},{ci_1[1]*100:+.1f}]%"
              f" dRset[{ci_s[0]:+.4f},{ci_s[1]:+.4f}]"
              f" dRrank[{ci_r[0]:+.4f},{ci_r[1]:+.4f}]  (dro-base, 95% CI)")

    hdr = f"{'eval / band':22}{'n':>6}"
    for nm in names:
        hdr += f" |  base   dro " if nm.startswith("Oracle") else f" |  base    dro"
    print(hdr)
    print("-" * 110)
    for ev in evals:
        show(ev, [k for k in keys if k[0] == ev])
    print("-" * 110)
    shifted = [k for k in keys if k[0] not in CONTROLS]
    show("OOD macro (shifted)", shifted)
    print("-" * 110)
    for l, band in LAYER_BAND.items():
        ks = [k for k in shifted if k[3] == l]
        if ks:
            show(f"shifted / {band}", ks)


if __name__ == "__main__":
    main()
