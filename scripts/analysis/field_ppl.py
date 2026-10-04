#!/usr/bin/env python3
"""Full-corpus PPL of one checkpoint on given Paloma fields.

Unlike the boundary probe (which measures loss only at selected low-margin
positions), this evaluates EVERY token of EVERY window in both the val and
test splits -- the unbiased perplexity of the checkpoint on that field.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Megatron-LM'))

from megatron.training.initialize import initialize_megatron
from megatron.training import get_args, get_model
from megatron.training.checkpointing import load_checkpoint

from routing_boundary_probe import (  # noqa: E402
    load_paloma_domain, load_indexed_docs, to_batch, per_position_loss,
    _positions, run_batches,
)


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--role", required=True)
    p.add_argument("--fields", nargs="+", required=True,
                   help="paloma domain ids, e.g. math.GN math.KT econ.EM")
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--max-docs", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", required=True)
    opt, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)

    initialize_megatron(
        args_defaults={"no_load_rng": True, "no_load_optim": True,
                       "exit_on_missing_checkpoint": True,
                       "micro_batch_size": opt.batch_size},
        ignore_unknown_args=True)
    args = get_args()
    from pretrain_gpt import model_provider
    model = get_model(model_provider, wrap_with_ddp=False)
    load_checkpoint(model, None, None)
    model = model[0]
    model.eval()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_model)

    # Per-split results kept separate: selecting fields on `val` and confirming
    # on `test` keeps a PPL-based field selection honest.
    results = {}
    if os.path.exists(opt.out):          # resume a partial run
        results = json.load(open(opt.out)).get("results", {})
        print(f"[{opt.role}] resuming: {len(results)} fields already done", flush=True)
    for fi, field in enumerate(opt.fields):
        if field in results:
            continue
        entry = {}
        for split in opt.splits:
            tot_nll, tot_tok, win_nll, win_uniq = 0.0, 0, [], []
            if field.startswith("dclm:"):
                # dclm:<shard-prefix>  -- in-distribution held-out; "split"
                # selects disjoint document ranges of the same shard.
                off = 0 if split == "val" else 4000
                rows, masks = load_indexed_docs(
                    field.split(":", 1)[1], off, tok, args.seq_length, opt.max_docs)
            else:
                cfg, dom = (field.split("|", 1) if "|" in field
                            else ("m2d2_s2orc_unsplit", field))
                rows, masks = load_paloma_domain(
                    cfg, dom, split, tok, args.seq_length, opt.max_docs)
            for i0, chunk in run_batches(rows, opt.batch_size):
                inp, tgt = to_batch(chunk)
                with torch.no_grad():
                    out = model(inp, _positions(inp), None)
                logits = out[0] if isinstance(out, tuple) else out
                loss = per_position_loss(logits, tgt)          # [b, s]
                for bi in range(inp.shape[0]):
                    nt = masks[i0 + bi]
                    tot_nll += float(loss[bi, :nt].sum())
                    tot_tok += nt
                    win_nll.append(round(float(loss[bi, :nt].mean()), 6))
                    # degeneracy: unique-token ratio flags junk windows
                    # (e.g. "q q q q ..." PDF-extraction artifacts)
                    ww = inp[bi, :nt]
                    win_uniq.append(round(float(ww.unique().numel() / max(nt, 1)), 4))
            if tot_tok == 0:
                # some Paloma field-splits are shorter than one window
                print(f"[{opt.role}] SKIP {field}/{split}: 0 windows", flush=True)
                continue
            entry[split] = {"nll": tot_nll / tot_tok, "ppl": math.exp(tot_nll / tot_tok),
                            "tokens": tot_tok, "windows": win_nll,
                            "uniq": win_uniq}
        if not entry:
            continue
        results[field] = entry
        msg = "  ".join(f"{s}: PPL={entry[s]['ppl']:.4f}" for s in opt.splits if s in entry)
        print(f"[{opt.role}] ({fi+1}/{len(opt.fields)}) {field:18} {msg}", flush=True)
        with open(opt.out, "w") as f:      # checkpoint after each field
            json.dump({"role": opt.role, "results": results}, f, indent=1)
    print(f"wrote {opt.out}", flush=True)


if __name__ == "__main__":
    main()
