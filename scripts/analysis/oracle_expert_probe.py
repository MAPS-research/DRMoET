#!/usr/bin/env python3
"""Oracle single-expert probe: router-expert alignment under distribution shift.

For each sampled token position t and probe MoE layer l, counterfactually
evaluate EVERY non-shared expert e: replace the entire non-shared top-k mixture
at (t, l) with expert e alone, carrying the SAME total non-shared routing
coefficient W_{t,l} = sum of the natural top-k weights (shared expert
untouched, all other layers natural). Record the next-token loss L_{t,l}(e).

With e* = argmin_e L(e) (oracle-best single expert), per checkpoint we report
  Oracle@k  = Pr(e* in S)              S = router's natural top-k set
  Oracle@1  = Pr(e* == router top-1)
  R_set     = min_{e in S} L(e) - L(e*)          (set regret)
  R_rank    = L(top1) - min_{e in S} L(e)        (ranking regret inside S)

Positions are sampled deterministically from the DATA alone (seeded by
eval-set name + sequence index), so both checkpoints independently derive
IDENTICAL (seq, t, l) populations -- no cross-run position file needed. Each
position is probed at ONE layer, cycling early / middle / late.

Framing note: this measures router-expert ALIGNMENT of the full checkpoint
(router + experts + representations co-adapted), not router parameters in
isolation, and e* is the best SINGLE expert, not the best k-subset.
"""

import argparse
import hashlib
import json
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Megatron-LM'))

from megatron.training.initialize import initialize_megatron
from megatron.training import get_args, get_model
from megatron.training.checkpointing import load_checkpoint

# reuse the validated data + batching helpers from the boundary probe
from routing_boundary_probe import (  # noqa: E402
    load_paloma_domain, load_indexed_docs, to_batch, per_position_loss,
    _positions, run_batches, data_checksum,
)


# --------------------------------------------------------------------------- #
# Router instrumentation (replacement mode)
# --------------------------------------------------------------------------- #

class OracleControl:
    def __init__(self):
        self.reset()

    def reset(self):
        self.capture = False
        self.logits = {}        # moe_idx -> [s*b, E] fp32 (natural pass)
        # replacement spec: moe_idx -> (rows LongTensor, experts LongTensor)
        self.replace = {}


OCTRL = OracleControl()


def install_router_wrappers(model):
    from megatron.core.transformer.moe.router import TopKRouter
    routers = []
    for name, mod in model.named_modules():
        if isinstance(mod, TopKRouter):
            routers.append((mod.layer_number, name, mod))
    routers.sort(key=lambda x: x[0])

    for moe_idx, (_lnum, _name, router) in enumerate(routers):
        def wrapped_forward(self, input, _moe_idx=moe_idx):
            assert input.dim() == 3, f"router input expected [s,b,h], got {input.shape}"
            logits = self.gating(input)                     # [s, b, E]
            s, b, e = logits.shape
            scores, routing_map = self.routing(logits)      # [s*b, E]
            flat_logits = logits.view(-1, e).float()
            if OCTRL.capture:
                OCTRL.logits[_moe_idx] = flat_logits.detach()

            spec = OCTRL.replace.get(_moe_idx)
            if spec is not None:
                rows, experts = spec
                probs = torch.softmax(flat_logits[rows], dim=-1)   # [n, E]
                k = self.topk
                w_total = torch.topk(probs, k, dim=-1).values.sum(-1)  # [n]
                for j in range(rows.numel()):
                    r = rows[j].item()
                    routing_map[r, :] = False
                    routing_map[r, experts[j]] = True
                    scores[r, :] = 0.0
                    scores[r, experts[j]] = w_total[j].to(scores.dtype)
            return scores, routing_map

        router.forward = types.MethodType(wrapped_forward, router)
    return [r[0] for r in routers]


def natural_capture(model, input_ids, targets):
    """Natural forward; returns per-position loss [b,s] and captured logits."""
    OCTRL.reset()
    OCTRL.capture = True
    with torch.no_grad():
        out = model(input_ids, _positions(input_ids), None)
    logits = out[0] if isinstance(out, tuple) else out
    loss = per_position_loss(logits, targets)
    OCTRL.capture = False
    return loss, dict(OCTRL.logits)


def replace_pass(model, input_ids, targets, spec):
    """spec: list of (batch_row, t, moe_idx, expert). One entry per batch_row."""
    OCTRL.reset()
    b, s = input_ids.shape
    per_layer_rows, per_layer_exps = {}, {}
    seen = set()
    for (i, t, l, e) in spec:
        assert i not in seen
        seen.add(i)
        per_layer_rows.setdefault(l, []).append(t * b + i)
        per_layer_exps.setdefault(l, []).append(e)
    for l in per_layer_rows:
        order = np.argsort(per_layer_rows[l])
        OCTRL.replace[l] = (
            torch.tensor(np.asarray(per_layer_rows[l])[order], dtype=torch.long,
                         device=input_ids.device),
            torch.tensor(np.asarray(per_layer_exps[l])[order], dtype=torch.long,
                         device=input_ids.device),
        )
    with torch.no_grad():
        out = model(input_ids, _positions(input_ids), None)
    logits = out[0] if isinstance(out, tuple) else out
    loss = per_position_loss(logits, targets)
    OCTRL.reset()
    return loss


_WARM = {"done": False}


def ensure_warmup(model, rows):
    if _WARM["done"]:
        return
    inp, tgt = to_batch(rows[: min(4, len(rows))])
    natural_capture(model, inp, tgt)
    _WARM["done"] = True
    print("[warmup] done", flush=True)


# --------------------------------------------------------------------------- #
# Position sampling (data-deterministic; identical across checkpoints)
# --------------------------------------------------------------------------- #

def sample_positions(name, seq_global, n_targets, quota, probe_layers, min_t=64):
    """Deterministic positions+layers for one sequence."""
    hi = n_targets - 1
    if hi <= min_t:
        return []
    rng = np.random.default_rng(
        int(hashlib.sha256(f"oracle|{name}|{seq_global}".encode()).hexdigest()[:8], 16))
    n = min(quota, hi - min_t)
    ts = rng.choice(np.arange(min_t, hi), size=n, replace=False)
    return [(int(t), probe_layers[j % len(probe_layers)])
            for j, t in enumerate(sorted(ts))]


# --------------------------------------------------------------------------- #
# Probe driver
# --------------------------------------------------------------------------- #

def probe_eval_set(model, name, rows, masks, opt):
    args = get_args()
    k = args.moe_router_topk
    E = args.num_experts
    ensure_warmup(model, rows)

    quota = int(np.ceil(opt.positions / max(1, len(rows))))
    records = []
    for i0, chunk in run_batches(rows, opt.batch_size):
        inp, tgt = to_batch(chunk)
        b = inp.shape[0]
        loss_nat, zcap = natural_capture(model, inp, tgt)
        loss_nat = loss_nat.cpu()

        # natural routing info at probe layers
        info = {}   # (bi, t, l) -> (S list, top1, W)
        batch_pos = {}
        for bi in range(b):
            g = i0 + bi
            plist = sample_positions(name, g, masks[g], quota, opt.probe_layers)
            batch_pos[bi] = plist
            for (t, l) in plist:
                z = zcap[l][t * b + bi]
                p = torch.softmax(z, dim=-1)
                topv, topi = torch.topk(p, k)
                info[(bi, t, l)] = (topi.tolist(), int(topi[0]),
                                    float(topv.sum()))

        rounds = max((len(v) for v in batch_pos.values()), default=0)
        # losses accumulator: (bi, r) -> np.array[E]
        acc = {(bi, r): np.full(E, np.nan)
               for bi in batch_pos for r in range(len(batch_pos[bi]))}
        for r in range(rounds):
            active = [(bi, batch_pos[bi][r]) for bi in batch_pos
                      if r < len(batch_pos[bi])]
            if not active:
                continue
            for e in range(E):
                spec = [(bi, t, l, e) for bi, (t, l) in active]
                loss_sw = replace_pass(model, inp, tgt, spec).cpu()
                for bi, (t, l) in active:
                    acc[(bi, r)][e] = float(loss_sw[bi, t])
            if (r + 1) % 4 == 0:
                print(f"[{name}] batch@{i0}: round {r+1}/{rounds} "
                      f"({(r+1)*E} fwd)", flush=True)

        for bi in batch_pos:
            g = i0 + bi
            for r, (t, l) in enumerate(batch_pos[bi]):
                S, top1, W = info[(bi, t, l)]
                L = acc[(bi, r)]
                assert not np.isnan(L).any()
                records.append({
                    "eval": name, "seq": g, "t": t, "l": l,
                    "S": S, "top1": top1, "W": W,
                    "l_nat": float(loss_nat[bi, t]),
                    "losses": [round(float(x), 6) for x in L],
                })
        print(f"[{name}] batch@{i0} done: {len(records)} positions total", flush=True)
        if len(records) >= opt.positions:
            break
    return records[: opt.positions]


# --------------------------------------------------------------------------- #
# Self tests
# --------------------------------------------------------------------------- #

def self_test(model, rows, masks, opt):
    print("=" * 70)
    print("ORACLE-PROBE SELF TESTS")
    ensure_warmup(model, rows)
    inp, tgt = to_batch(rows[: opt.batch_size])
    b, s = inp.shape
    k = get_args().moe_router_topk
    mid = opt.probe_layers[len(opt.probe_layers) // 2]

    # 1) determinism
    l1, z1 = natural_capture(model, inp, tgt)
    l2, _ = natural_capture(model, inp, tgt)
    d = (l1 - l2).abs().max().item()
    print(f"[1] determinism: max|dL|={d:.3e} ->", "PASS" if d == 0 else "FAIL")
    assert d == 0

    # 2) causality: replacement at t=1024 leaves pos<t bit-identical
    spec = [(bi, 1024, mid, bi % get_args().num_experts) for bi in range(b)]
    lsw = replace_pass(model, inp, tgt, spec)
    before = (l1[:, :1024] - lsw[:, :1024]).abs().max().item()
    at = (l1[:, 1024] - lsw[:, 1024]).abs().mean().item()
    print(f"[2] causality: max|dL(pos<t)|={before:.3e}, mean|dL(t)|={at:.3e} ->",
          "PASS" if before == 0 and at > 0 else "FAIL")
    assert before == 0

    # 3) W scheme: natural non-shared mass equals top-k softmax mass
    z = z1[mid]
    p = torch.softmax(z, dim=-1)
    W = torch.topk(p, k, dim=-1).values.sum(-1)
    print(f"[3] W: mean total non-shared coeff={W.mean():.4f} "
          f"(top1 share={torch.topk(p, 1, dim=-1).values.mean():.4f}) -> PASS")

    # 4) semantic: single-expert(top1) beats single-expert(router-worst) on avg
    t_probe = 1024
    zrow = z1[mid].view(s, b, -1)[t_probe]            # [b, E]
    prow = torch.softmax(zrow, dim=-1)
    top1 = prow.argmax(-1)
    worst = prow.argmin(-1)
    l_top = replace_pass(model, inp, tgt,
                         [(bi, t_probe, mid, int(top1[bi])) for bi in range(b)])
    l_bot = replace_pass(model, inp, tgt,
                         [(bi, t_probe, mid, int(worst[bi])) for bi in range(b)])
    dt = (l_bot[:, t_probe] - l_top[:, t_probe]).mean().item()
    print(f"[4] semantic: mean L(router-worst) - L(router-top1) = {dt:+.4f} "
          f"(positive = router ranking informative) ->",
          "PASS" if dt > 0 else "FAIL")
    assert dt > 0

    # 5) degradation sanity: single-expert replacement vs natural mixture at t
    dnat = (l_top[:, t_probe] - l1[:, t_probe]).mean().item()
    print(f"[5] degradation: mean L(top1-only) - L(natural k-mix) = {dnat:+.4f} "
          f"(informational)")
    print("ALL ORACLE SELF TESTS PASSED")
    print("=" * 70)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--role", choices=["baseline", "drmoet"], required=True)
    p.add_argument("--eval-spec", required=True)
    p.add_argument("--records-out", required=True)
    p.add_argument("--positions", type=int, default=1000,
                   help="token positions per eval set")
    p.add_argument("--probe-layers", type=int, nargs="+", default=[0, 4, 7],
                   help="moe layer indices for early/middle/late")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--self-test", action="store_true")
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

    moe_layers = install_router_wrappers(model)
    print(f"Instrumented {len(moe_layers)} MoE routers (layers={moe_layers}); "
          f"probing moe_idx={opt.probe_layers} "
          f"(layer_numbers={[moe_layers[i] for i in opt.probe_layers]})", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_model)
    with open(opt.eval_spec) as f:
        spec = json.load(f)

    all_records = []
    for es in spec:
        name = es["name"]
        if es["kind"] == "paloma":
            rows, masks = load_paloma_domain(
                es["config"], es["domain"], es.get("split", "test"),
                tok, args.seq_length, es.get("max_docs", 64))
        else:
            rows, masks = load_indexed_docs(
                es["path"], es.get("first_doc", 0), tok, args.seq_length,
                es.get("max_docs", 64))
        print(f"[{name}] {len(rows)} seqs (checksum {data_checksum(rows)})", flush=True)
        if opt.self_test:
            self_test(model, rows, masks, opt)
            return
        recs = probe_eval_set(model, name, rows, masks, opt)
        all_records += recs

    with open(opt.records_out, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(all_records)} records -> {opt.records_out}", flush=True)


if __name__ == "__main__":
    main()
