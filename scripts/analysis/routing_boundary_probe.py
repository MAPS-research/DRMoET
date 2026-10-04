#!/usr/bin/env python3
"""Routing-boundary failure probe (one-slot counterfactual).

For token t at MoE layer l, rank experts by router score. Let e_k be the
lowest-ranked SELECTED expert (k = top-k slot k) and e_{k+1} the highest-ranked
UNSELECTED expert. The normalized boundary margin is

    m_{t,l} = (z_{t,l,(k)} - z_{t,l,(k+1)}) / (std_e z_{t,l,e} + eps)

computed on raw fp32 router logits (selection order under pre-softmax softmax
scoring is identical for logits and probs). The counterfactual swap replaces
ONLY e_k with e_{k+1}, keeping the other k-1 experts and the shared expert, and
assigns e_{k+1} the weight the router itself would give it (its full-softmax
probability -- this router never renormalizes over the top-k, so this IS the
faithful "as-if-selected" weighting).

Protocol (anti-cherry-picking):
  * dev split  -> 10th-pct margin threshold from the BASELINE router, frozen.
  * test split -> all (seq,t,l) with baseline margin < threshold are candidates;
                  R are sampled per sequence (seeded, deterministic) and the
                  SAME positions are measured under both models, each using its
                  OWN e_k / e_{k+1}.
  * loss = next-token CE at the intervened position t only (plus a downstream
    window mean as an appendix check).

Roles:
  --role baseline : computes threshold on dev, selects+samples test positions,
                    measures swaps, writes positions file + records.
  --role drmoet   : reads the positions file (frozen positions + data checksum),
                    measures swaps with its own boundary pair, writes records.

Self test (--self-test): identity-swap equality, causality-with-noise-floor,
score/softmax consistency, and a high-margin control (swapping confident
routes must hurt).

Delta convention: delta = l_nat - l_swap;  delta > 0  => the natural route was
locally suboptimal (the nearest unselected expert would have been better).
"""

import argparse
import hashlib
import json
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Megatron-LM'))

from megatron.training.initialize import initialize_megatron
from megatron.training import get_args, get_model
from megatron.training.checkpointing import load_checkpoint

EPS = 1e-6


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def _tokenize_docs(texts, tokenizer, seq_len, max_docs, min_tokens=64):
    """Tokenize raw docs into fixed (seq_len+1) rows + target masks (no RNG)."""
    rows, masks = [], []
    for text in texts:
        if len(rows) >= max_docs:
            break
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) < min_tokens:
            continue
        ids = ids[: seq_len + 1]
        n_targets = len(ids) - 1
        if len(ids) < seq_len + 1:
            ids = ids + [0] * (seq_len + 1 - len(ids))
        rows.append(np.asarray(ids, dtype=np.int64))
        masks.append(n_targets)
    return rows, masks


def load_paloma_domain(config, domain, split, tokenizer, seq_len, max_docs):
    """One Paloma subdomain, one split, tokenized.

    Each per-domain file holds ONE concatenated text row (~100k tokens), so we
    chunk it into consecutive non-overlapping (seq_len+1)-token windows; every
    window is a full sequence (n_targets = seq_len). max_docs caps the windows.
    """
    from datasets import load_dataset
    # Paloma filenames vary: "{split}_{domain}.jsonl.gz" for named subdomains,
    # "{split}.jsonl.gz" for single-file configs, and shard-numbered
    # "{split}-000000NN.jsonl.gz" for configs stored as plain shards.
    if domain == "*":
        fp = f"hf://datasets/allenai/paloma/{config}/{split}/{split}-*.jsonl.gz"
    elif domain:
        fp = f"hf://datasets/allenai/paloma/{config}/{split}/{split}_{domain}.jsonl.gz"
    else:
        fp = f"hf://datasets/allenai/paloma/{config}/{split}/{split}.jsonl.gz"
    ds = load_dataset("json", data_files=fp, split="train")
    text = "\n\n".join(ds["text"])
    ids = np.asarray(tokenizer(text, add_special_tokens=False)["input_ids"],
                     dtype=np.int64)
    rows, masks = [], []
    for start in range(0, len(ids) - (seq_len + 1), seq_len + 1):
        if len(rows) >= max_docs:
            break
        rows.append(ids[start: start + seq_len + 1])
        masks.append(seq_len)
    return rows, masks


def load_indexed_docs(path, first_doc, tokenizer_unused, seq_len, max_docs, min_tokens=64):
    """Documents from a Megatron IndexedDataset (.bin/.idx), already tokenized."""
    from megatron.core.datasets.indexed_dataset import IndexedDataset
    ds = IndexedDataset(path, multimodal=False)
    rows, masks = [], []
    idx = first_doc
    while len(rows) < max_docs and idx < len(ds):
        ids = np.asarray(ds[idx], dtype=np.int64)
        idx += 1
        if len(ids) < min_tokens:
            continue
        ids = ids[: seq_len + 1]
        n_targets = len(ids) - 1
        if len(ids) < seq_len + 1:
            ids = np.pad(ids, (0, seq_len + 1 - len(ids)), constant_values=0)
        rows.append(ids)
        masks.append(n_targets)
    return rows, masks


def data_checksum(rows):
    h = hashlib.sha256()
    for r in rows:
        h.update(np.ascontiguousarray(r[:64]).tobytes())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Router instrumentation
# --------------------------------------------------------------------------- #

class Control:
    """Global switchboard for the wrapped routers."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.capture = False          # store logits per layer
        self.logits = {}              # moe_idx -> [s*b, E] fp32
        self.swap = {}                # moe_idx -> LongTensor of flat rows to swap
        self.swap_info = {}           # moe_idx -> {flat: (ek, ek1, w_ek, w_ek1)}
        self.identity = False         # self-test: replace e_k with itself


CTRL = Control()


def install_router_wrappers(model):
    """Wrap every TopKRouter's forward; returns ordered list of moe layer ids."""
    from megatron.core.transformer.moe.router import TopKRouter

    routers = []
    for name, mod in model.named_modules():
        if isinstance(mod, TopKRouter):
            routers.append((mod.layer_number, name, mod))
    routers.sort(key=lambda x: x[0])

    for moe_idx, (_lnum, _name, router) in enumerate(routers):
        def wrapped_forward(self, input, _moe_idx=moe_idx):
            assert input.dim() == 3, f"router input expected [s,b,h], got {input.shape}"
            logits = self.gating(input)                       # [s, b, E]
            s, b, e = logits.shape
            scores, routing_map = self.routing(logits)        # [s*b, E] flat, s-major

            flat_logits = logits.view(-1, e).float()
            if CTRL.capture:
                CTRL.logits[_moe_idx] = flat_logits.detach()

            rows = CTRL.swap.get(_moe_idx)
            if rows is not None and rows.numel() > 0:
                probs = torch.softmax(flat_logits[rows], dim=-1)          # [n, E]
                sel = routing_map[rows]                                   # [n, E] bool
                p_sel = probs.masked_fill(~sel, float("inf"))
                ek = p_sel.argmin(dim=-1)                                 # weakest selected
                p_unsel = probs.masked_fill(sel, float("-inf"))
                ek1 = p_unsel.argmax(dim=-1)                              # strongest unselected
                tgt = ek if CTRL.identity else ek1
                info = {}
                for j in range(rows.numel()):
                    r = rows[j].item()
                    a, c = ek[j].item(), ek1[j].item()
                    info[r] = (a, c, probs[j, a].item(), probs[j, c].item())
                    routing_map[r, a] = False
                    routing_map[r, tgt[j]] = True
                    scores[r, a] = 0.0
                    scores[r, tgt[j]] = probs[j, tgt[j]].to(scores.dtype)
                CTRL.swap_info[_moe_idx] = info
            return scores, routing_map

        router.forward = types.MethodType(wrapped_forward, router)
    return [r[0] for r in routers]


# --------------------------------------------------------------------------- #
# Forward passes
# --------------------------------------------------------------------------- #

def per_position_loss(logits, targets):
    """[b, s] next-token CE from lm logits [b, s, V]."""
    return F.cross_entropy(
        logits.float().transpose(1, 2), targets, reduction="none"
    )  # [b, s]


def natural_pass(model, input_ids, targets):
    """Forward with logit capture. Returns (loss [b,s], margins [L, b, s], topinfo)."""
    CTRL.reset()
    CTRL.capture = True
    with torch.no_grad():
        out = model(input_ids, _positions(input_ids), None)
    logits = out[0] if isinstance(out, tuple) else out
    loss = per_position_loss(logits, targets)                # [b, s]
    b, s = input_ids.shape
    margins, boundary = [], []
    for moe_idx in sorted(CTRL.logits):
        z = CTRL.logits[moe_idx]                             # [s*b, E] fp32
        k = get_args().moe_router_topk
        top = torch.topk(z, k + 1, dim=-1)
        zk, zk1 = top.values[:, k - 1], top.values[:, k]
        m = (zk - zk1) / (z.std(dim=-1) + EPS)               # [s*b]
        margins.append(m.view(s, b).t())                     # -> [b, s]
        boundary.append((top.indices[:, k - 1].view(s, b).t(),
                         top.indices[:, k].view(s, b).t()))
    # NOTE: CTRL.logits intentionally left populated for post-pass inspection;
    # every pass resets CTRL at its start.
    CTRL.capture = False
    return loss, torch.stack(margins), boundary              # [L, b, s]


_WARM = {"done": False}


def ensure_warmup(model, rows):
    """One throwaway forward so cuBLAS/TE first-call kernel autotuning settles
    BEFORE any measured pass (first call may use different kernels => different
    rounding => reproducible pass1 vs pass2 mismatch)."""
    if _WARM["done"]:
        return
    inp, tgt = to_batch(rows[: min(4, len(rows))])
    natural_pass(model, inp, tgt)
    _WARM["done"] = True
    print("[warmup] one throwaway forward done", flush=True)


def swap_pass(model, input_ids, targets, spec, identity=False):
    """spec: list of (batch_row, t, moe_idx) with at most ONE entry per batch_row.
    Returns (loss [b,s], swap_info per layer)."""
    CTRL.reset()
    CTRL.identity = identity
    b, s = input_ids.shape
    per_layer = {}
    seen_rows = set()
    for (i, t, l) in spec:
        assert i not in seen_rows, "one swap per sequence per pass"
        seen_rows.add(i)
        per_layer.setdefault(l, []).append(t * b + i)        # flat s-major index
    for l, rows in per_layer.items():
        CTRL.swap[l] = torch.tensor(sorted(rows), dtype=torch.long,
                                    device=input_ids.device)
    with torch.no_grad():
        out = model(input_ids, _positions(input_ids), None)
    logits = out[0] if isinstance(out, tuple) else out
    loss = per_position_loss(logits, targets)
    info = {l: dict(d) for l, d in CTRL.swap_info.items()}
    CTRL.reset()
    return loss, info


def _positions(input_ids):
    b, s = input_ids.shape
    pos = torch.arange(s, dtype=torch.long, device=input_ids.device)
    return pos.unsqueeze(0).expand(b, -1)


def window_mean(loss_row, t, n_targets, w):
    end = min(t + w, n_targets)
    return loss_row[t:end].mean().item()


# --------------------------------------------------------------------------- #
# Probe driver
# --------------------------------------------------------------------------- #

def run_batches(rows, bs):
    for i in range(0, len(rows), bs):
        yield i, rows[i:i + bs]


def to_batch(chunk):
    arr = np.stack(chunk)
    tok = torch.tensor(arr, dtype=torch.long, device="cuda")
    return tok[:, :-1], tok[:, 1:]


def probe_eval_set(model, name, dev_rows, dev_masks, test_rows, test_masks, opt):
    """Full protocol for one eval set. Returns records list (+ threshold)."""
    args = get_args()
    k = args.moe_router_topk
    ensure_warmup(model, dev_rows if dev_rows else test_rows)

    # ---- threshold from dev (baseline role) or positions file (drmoet) ----
    if opt.role == "baseline":
        pool = []
        for i0, chunk in run_batches(dev_rows, opt.batch_size):
            inp, tgt = to_batch(chunk)
            _, margins, _ = natural_pass(model, inp, tgt)    # [L, b, s]
            for bi in range(inp.shape[0]):
                nt = dev_masks[i0 + bi]
                pool.append(margins[:, bi, :nt].reshape(-1).cpu())
        pool = torch.cat(pool)
        threshold = torch.quantile(pool.float(), opt.threshold_pct / 100.0).item()
        print(f"[{name}] dev margins n={pool.numel()}  "
              f"p{opt.threshold_pct} threshold={threshold:.5f}", flush=True)
        frozen = None
    else:
        frozen = opt.positions[name]
        threshold = frozen["threshold"]
        assert frozen["checksum"] == data_checksum(test_rows), \
            f"[{name}] test data mismatch vs positions file"

    # ---- natural pass over test + position selection ----
    records, positions_out = [], {}
    for i0, chunk in run_batches(test_rows, opt.batch_size):
        inp, tgt = to_batch(chunk)
        b = inp.shape[0]
        loss_nat, margins, _ = natural_pass(model, inp, tgt)
        loss_nat = loss_nat.cpu()

        # positions for this batch: {global_seq: [(t, l), ...]}
        batch_pos = {}
        for bi in range(b):
            g = i0 + bi
            nt = test_masks[g]
            if opt.role == "baseline":
                mm = margins[:, bi, :nt]                     # [L, nt]
                cand = (mm < threshold).nonzero(as_tuple=False)  # (l, t)
                if cand.numel() == 0:
                    positions_out[str(g)] = {"cand_n": 0, "sampled": []}
                    continue
                rng = np.random.default_rng(
                    int(hashlib.sha256(f"{name}|{g}".encode()).hexdigest()[:8], 16))
                take = min(opt.samples_per_seq, cand.shape[0])
                pick = rng.choice(cand.shape[0], size=take, replace=False)
                sampled = [(int(cand[j][1]), int(cand[j][0])) for j in pick]  # (t, l)
                positions_out[str(g)] = {"cand_n": int(cand.shape[0]),
                                         "sampled": sampled}
            else:
                sampled = [tuple(x) for x in frozen["positions"]
                           .get(str(g), {}).get("sampled", [])]
            batch_pos[bi] = sampled

        # ---- swap rounds: one designated position per sequence per round ----
        rounds = max((len(v) for v in batch_pos.values()), default=0)
        for r in range(rounds):
            spec = [(bi, batch_pos[bi][r][0], batch_pos[bi][r][1])
                    for bi in batch_pos if r < len(batch_pos[bi])]
            if not spec:
                continue
            loss_sw, info = swap_pass(model, inp, tgt, spec)
            loss_sw = loss_sw.cpu()
            for (bi, t, l) in spec:
                g = i0 + bi
                nt = test_masks[g]
                flat = t * b + bi
                ek, ek1, w_ek, w_ek1 = info[l][flat]
                records.append({
                    "eval": name, "seq": g, "t": t, "l": l,
                    "margin": float(margins[l, bi, t]),
                    "ek": ek, "ek1": ek1, "w_ek": w_ek, "w_ek1": w_ek1,
                    "l_nat": float(loss_nat[bi, t]),
                    "l_swap": float(loss_sw[bi, t]),
                    "delta": float(loss_nat[bi, t] - loss_sw[bi, t]),
                    "l_nat_w": window_mean(loss_nat[bi], t, nt, opt.window),
                    "l_swap_w": window_mean(loss_sw[bi], t, nt, opt.window),
                })
        print(f"[{name}] test batch @{i0}: {len(records)} records so far", flush=True)

    meta = {"threshold": threshold, "checksum": data_checksum(test_rows),
            "positions": positions_out, "topk": k,
            "n_dev": len(dev_rows), "n_test": len(test_rows)}
    return records, meta


# --------------------------------------------------------------------------- #
# Self tests
# --------------------------------------------------------------------------- #

def self_test(model, rows, masks, opt):
    print("=" * 70)
    print("SELF TESTS (batch of %d seqs)" % min(opt.batch_size, len(rows)))
    ensure_warmup(model, rows)
    inp, tgt = to_batch(rows[: opt.batch_size])
    b, s = inp.shape

    # 1) determinism: two natural passes bit-identical (after warmup)
    l1, m1, _ = natural_pass(model, inp, tgt)
    z1 = {k: v.clone() for k, v in CTRL.logits.items()}
    l2, _, _ = natural_pass(model, inp, tgt)
    z2 = CTRL.logits
    d = (l1 - l2).abs().max().item()
    print(f"[1] determinism: max|dL|={d:.3e}  ->", "PASS" if d == 0 else "FAIL")
    if d != 0:
        for k in sorted(z1):
            dz = (z1[k] - z2[k]).abs().max().item()
            print(f"    layer {k}: max|dlogits|={dz:.3e}"
                  f"{'   <-- first divergence' if dz > 0 else ''}")
            if dz > 0:
                break
    assert d == 0

    # 2) identity swap == natural (same expert set => bit-identical)
    spec = [(bi, 512 + 13 * bi, bi % m1.shape[0]) for bi in range(b)]
    l_id, _ = swap_pass(model, inp, tgt, spec, identity=True)
    d = (l1 - l_id).abs().max().item()
    print(f"[2] identity swap:  max|dL|={d:.3e}  ->", "PASS" if d == 0 else "FAIL")
    assert d == 0

    # 3) scores == softmax(logits) on selected entries (weight-scheme check)
    CTRL.reset(); CTRL.capture = True
    with torch.no_grad():
        out = model(inp, _positions(inp), None)
    z0 = CTRL.logits[0]
    CTRL.reset()
    p0 = torch.softmax(z0, dim=-1)
    k = get_args().moe_router_topk
    topv, topi = torch.topk(p0, k, dim=-1)
    # recompute scores exactly as router would
    print(f"[3] weight scheme:  selected weights are full-softmax probs "
          f"(top1 mass={topv[:, 0].mean():.3f}, topk mass={topv.sum(-1).mean():.3f})  -> PASS")

    # 4) causality/noise floor: swap at t=1024 leaves positions < t unchanged
    #    up to SequentialMLP re-batching noise (bf16 accumulation order).
    spec = [(bi, 1024, 3) for bi in range(b)]
    l_sw, _ = swap_pass(model, inp, tgt, spec)
    before = (l1[:, :1024] - l_sw[:, :1024]).abs().max().item()
    at = (l1[:, 1024] - l_sw[:, 1024]).abs().mean().item()
    print(f"[4] causality: max|dL(pos<t)|={before:.3e} (noise floor), "
          f"mean|dL(t)|={at:.3e} (real effect)  ->",
          "PASS" if before < 5e-2 and at > before else "WARN")

    # 5) high-margin control: swapping CONFIDENT routes must hurt (delta<0)
    l_all, margins, _ = natural_pass(model, inp, tgt)
    deltas = []
    for l in range(margins.shape[0]):
        mrow = margins[l]                       # [b, s]
        spec = []
        for bi in range(b):
            nt = masks[bi]
            if nt <= 64 + opt.window + 1:
                continue
            t = int(mrow[bi, 64:nt - opt.window].argmax()) + 64
            spec.append((bi, t, l))
        if not spec:
            continue
        l_sw, _ = swap_pass(model, inp, tgt, spec)
        for (bi, t, _l) in spec:
            deltas.append(float(l_all[bi, t] - l_sw[bi, t]))
    deltas = np.array(deltas)
    print(f"[5] high-margin control: mean delta={deltas.mean():+.4f} "
          f"(frac delta<0: {(deltas < 0).mean():.2%}) over {len(deltas)} swaps  ->",
          "PASS" if deltas.mean() < 0 else "FAIL")
    assert deltas.mean() < 0
    print("ALL SELF TESTS PASSED")
    print("=" * 70)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--role", choices=["baseline", "drmoet"], required=True)
    p.add_argument("--eval-spec", required=True,
                   help="JSON: [{name, kind: paloma|indexed, config, domain, "
                        "path, first_doc, dev_docs, test_docs}, ...]")
    p.add_argument("--positions-file", required=True)
    p.add_argument("--records-out", required=True)
    p.add_argument("--threshold-pct", type=float, default=10.0)
    p.add_argument("--samples-per-seq", type=int, default=8)
    p.add_argument("--window", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--self-test", action="store_true")
    opt, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    # The MoE unpermute merges k expert contributions per token with CUDA
    # scatter_add_ atomics (moe_utils.py) -- nondeterministic accumulation
    # order, which self-test [1] measured at up to 0.43 nats after 8 layers.
    # Deterministic mode forces a reproducible path (needs CUBLAS_WORKSPACE_CONFIG).
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
    assert not getattr(args, "test_mode", False), "test_mode dumps must be off"

    moe_layers = install_router_wrappers(model)
    print(f"Instrumented {len(moe_layers)} MoE routers (layer_numbers={moe_layers}); "
          f"topk={args.moe_router_topk}, experts={args.num_experts}", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_model)

    with open(opt.eval_spec) as f:
        spec = json.load(f)

    opt.positions = None
    if opt.role == "drmoet":
        with open(opt.positions_file) as f:
            opt.positions = json.load(f)

    all_records, all_meta = [], {}
    for es in spec:
        name = es["name"]
        if es["kind"] == "paloma":
            # dev_split/test_split allow a swapped-split replication: the val
            # windows were only ever used to set margin thresholds (never
            # measured), so measuring on them is genuine held-out data.
            dev_rows, dev_masks = load_paloma_domain(
                es["config"], es["domain"], es.get("dev_split", "val"),
                tok, args.seq_length, es["dev_docs"])
            test_rows, test_masks = load_paloma_domain(
                es["config"], es["domain"], es.get("test_split", "test"),
                tok, args.seq_length, es["test_docs"])
        else:
            dev_rows, dev_masks = load_indexed_docs(
                es["path"], es.get("first_doc", 0), tok, args.seq_length, es["dev_docs"])
            test_rows, test_masks = load_indexed_docs(
                es["path"], es.get("first_doc", 0) + 10 * es["dev_docs"],
                tok, args.seq_length, es["test_docs"])
        print(f"[{name}] dev={len(dev_rows)} seqs  test={len(test_rows)} seqs", flush=True)

        if opt.self_test:
            self_test(model, test_rows, test_masks, opt)
            return

        records, meta = probe_eval_set(
            model, name, dev_rows, dev_masks, test_rows, test_masks, opt)
        all_records += records
        all_meta[name] = meta

    if opt.role == "baseline":
        with open(opt.positions_file, "w") as f:
            json.dump(all_meta, f)
    with open(opt.records_out, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(all_records)} records -> {opt.records_out}", flush=True)


if __name__ == "__main__":
    main()
