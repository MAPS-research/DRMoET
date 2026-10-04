#!/usr/bin/env python3
"""Summarize routing-boundary probe records into the rebuttal table.

Metrics per eval set (positions frozen from the BASELINE's low-margin dev
threshold; identical (seq,t,l) populations for both models):

  FailureRate  = Pr(delta > tau)          delta = l_nat - l_swap
  MeanRegret   = E[max(delta, 0)]
  RepairRate   = Pr(delta_drmoet <= tau | delta_baseline > tau)
  OutcomeRepair= Pr(l_nat_drmoet <= l_swap_baseline | delta_baseline > tau)

tau=0 is the literal definition; tau=0.01 nats is a noise-robust variant (the
SequentialMLP re-batching bf16 noise floor sits well below 0.01). Both are
reported. 95% CIs by bootstrap over SEQUENCES (not positions), which respects
within-sequence correlation.

Usage:
  python summarize_boundary.py --dir analysis/routing_boundary [--tau 0.0]
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def key(r):
    return (r["eval"], r["seq"], r["t"], r["l"])


def boot_ci(per_seq_values, stat, n=2000, seed=0):
    """Bootstrap over sequences; per_seq_values: {seq: [records]}."""
    seqs = list(per_seq_values)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.choice(len(seqs), size=len(seqs), replace=True)
        pooled = [r for i in pick for r in per_seq_values[seqs[i]]]
        if pooled:
            vals.append(stat(pooled))
    return np.percentile(vals, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="analysis/routing_boundary")
    ap.add_argument("--taus", type=float, nargs="+", default=[0.0, 0.01])
    args = ap.parse_args()

    base = {key(r): r for r in load(os.path.join(args.dir, "records_baseline.jsonl"))}
    dro = {key(r): r for r in load(os.path.join(args.dir, "records_drmoet.jsonl"))}
    joint_keys = sorted(set(base) & set(dro))
    only_b, only_d = len(base) - len(joint_keys), len(dro) - len(joint_keys)
    print(f"records: baseline={len(base)} drmoet={len(dro)} joined={len(joint_keys)}"
          f" (unmatched: {only_b}/{only_d})\n")

    evals = sorted({k[0] for k in joint_keys})
    for tau in args.taus:
        print("=" * 100)
        print(f"tau = {tau}  (failure := delta > tau)")
        hdr = (f"{'eval set':16}{'n':>6}{'base fail%':>11}{'dro fail%':>11}"
               f"{'rel.red%':>9}{'base regret':>12}{'dro regret':>12}"
               f"{'repair%':>9}{'outcome%':>10}")
        print(hdr)
        macro = defaultdict(list)
        for ev in evals:
            keys = [k for k in joint_keys if k[0] == ev]
            per_seq = defaultdict(list)
            for k in keys:
                per_seq[k[1]].append((base[k], dro[k]))
            pooled = [p for v in per_seq.values() for p in v]
            db = np.array([b["delta"] for b, _ in pooled])
            dd = np.array([d["delta"] for _, d in pooled])
            fb, fd = (db > tau).mean(), (dd > tau).mean()
            rb, rd = np.maximum(db, 0).mean(), np.maximum(dd, 0).mean()
            failed = db > tau
            repair = (dd[failed] <= tau).mean() if failed.any() else float("nan")
            outcome = np.mean([d["l_nat"] <= b["l_swap"]
                               for (b, d), f in zip(pooled, failed) if f]) \
                if failed.any() else float("nan")
            rel = (fb - fd) / fb * 100 if fb > 0 else float("nan")
            ci_b = boot_ci(per_seq, lambda P: np.mean([b["delta"] > tau for b, _ in P]))
            ci_d = boot_ci(per_seq, lambda P: np.mean([d["delta"] > tau for _, d in P]))
            print(f"{ev:16}{len(pooled):>6}{fb*100:>10.2f}%{fd*100:>10.2f}%"
                  f"{rel:>8.1f}%{rb:>12.4f}{rd:>12.4f}"
                  f"{repair*100:>8.1f}%{outcome*100:>9.1f}%")
            print(f"{'':16}{'':>6}  [{ci_b[0]*100:5.2f},{ci_b[1]*100:5.2f}]"
                  f" [{ci_d[0]*100:5.2f},{ci_d[1]*100:5.2f}]   (95% CI, seq bootstrap)")
            if ev != "dclm":
                macro["fb"].append(fb); macro["fd"].append(fd)
                macro["rb"].append(rb); macro["rd"].append(rd)
                macro["rep"].append(repair); macro["out"].append(outcome)
        if macro["fb"]:
            fb, fd = np.mean(macro["fb"]), np.mean(macro["fd"])
            print(f"{'OOD macro avg':16}{'':>6}{fb*100:>10.2f}%{fd*100:>10.2f}%"
                  f"{(fb-fd)/fb*100 if fb>0 else float('nan'):>8.1f}%"
                  f"{np.mean(macro['rb']):>12.4f}{np.mean(macro['rd']):>12.4f}"
                  f"{np.nanmean(macro['rep'])*100:>8.1f}%{np.nanmean(macro['out'])*100:>9.1f}%")
        print()

    # margin sanity: distribution of margins at measured positions
    mb = np.array([base[k]["margin"] for k in joint_keys])
    md = np.array([dro[k]["margin"] for k in joint_keys])
    print(f"margins at frozen positions: baseline median={np.median(mb):.4f}  "
          f"drmoet median={np.median(md):.4f} (own-router margins; drmoet not "
          f"threshold-selected, so its margins need not be small)")


if __name__ == "__main__":
    main()
