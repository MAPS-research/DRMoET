#!/usr/bin/env python3
"""Compute per-window degeneracy stats for Paloma fields (CPU only).

For each (config, domain, split) it re-derives the same (seq_len+1)-token
windows the PPL sweep used and records, per window:
  uniq  - unique-token ratio (low => repetitive/degenerate)
  top1  - frequency of the single most common token
  ent   - token-distribution entropy (nats)

Joined against the PPL records this lets us STRATIFY by degeneracy instead of
excluding windows, so we can ask whether DRMoET's advantage on low-entropy
sequences is systematic rather than a few outliers.
"""
import json
import sys
from collections import Counter

import numpy as np
from transformers import AutoTokenizer

SEQ = 2048


def main():
    fields = sys.argv[1].split()
    out = sys.argv[2]
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-12b")
    from datasets import load_dataset
    res = {}
    for i, field in enumerate(fields):
        cfg, dom = field.split("|", 1) if "|" in field else ("m2d2_s2orc_unsplit", field)
        res[field] = {}
        for split in ("val", "test"):
            suffix = f"_{dom}" if dom else ""
            fp = (f"hf://datasets/allenai/paloma/{cfg}/{split}/"
                  f"{split}{suffix}.jsonl.gz")
            ds = load_dataset("json", data_files=fp, split="train")
            ids = np.asarray(tok("\n\n".join(ds["text"]),
                                 add_special_tokens=False)["input_ids"],
                             dtype=np.int64)
            stats = []
            for s in range(0, len(ids) - (SEQ + 1), SEQ + 1):
                if len(stats) >= 64:
                    break
                w = ids[s:s + SEQ + 1][:SEQ]
                c = Counter(w.tolist())
                n = len(w)
                p = np.array(list(c.values()), dtype=float) / n
                stats.append({"uniq": round(len(c) / n, 4),
                              "top1": round(max(c.values()) / n, 4),
                              "ent": round(float(-(p * np.log(p)).sum()), 4)})
            res[field][split] = stats
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(fields)}", flush=True)
            json.dump(res, open(out, "w"))
    json.dump(res, open(out, "w"))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
