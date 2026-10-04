import json, os, sys, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..','..','Megatron-LM'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from megatron.training.initialize import initialize_megatron
from megatron.training import get_args, get_model
from megatron.training.checkpointing import load_checkpoint
from routing_boundary_probe import load_paloma_domain, to_batch, per_position_loss, _positions, run_batches
import argparse
p=argparse.ArgumentParser(add_help=False)
p.add_argument("--role",required=True); p.add_argument("--fields",nargs="+",required=True)
p.add_argument("--out",required=True); p.add_argument("--batch-size",type=int,default=16)
o,rem=p.parse_known_args(); sys.argv=[sys.argv[0]]+rem
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8"); torch.use_deterministic_algorithms(True)
initialize_megatron(args_defaults={"no_load_rng":True,"no_load_optim":True,
    "exit_on_missing_checkpoint":True,"micro_batch_size":o.batch_size}, ignore_unknown_args=True)
a=get_args()
from pretrain_gpt import model_provider
m=get_model(model_provider,wrap_with_ddp=False); load_checkpoint(m,None,None); m=m[0]; m.eval()
from transformers import AutoTokenizer
tk=AutoTokenizer.from_pretrained(a.tokenizer_model)
res={}
for f in o.fields:
    res[f]={}
    for sp in ("val","test"):
        rows,masks=load_paloma_domain("m2d2_s2orc_unsplit",f,sp,tk,a.seq_length,64)
        w=[]
        for i0,ch in run_batches(rows,o.batch_size):
            inp,tgt=to_batch(ch)
            with torch.no_grad(): out=m(inp,_positions(inp),None)
            lg=out[0] if isinstance(out,tuple) else out
            L=per_position_loss(lg,tgt)
            for bi in range(inp.shape[0]):
                nt=masks[i0+bi]; w.append(float(L[bi,:nt].mean()))
        res[f][sp]=w
        print(f"[{o.role}] {f} {sp}: {len(w)} windows",flush=True)
json.dump(res,open(o.out,"w"))
