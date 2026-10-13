import sys; sys.path.insert(0,'/home/claude/code')
import numpy as np, json
from gpt2np import GPT2
g=GPT2(); ids=np.load("/home/claude/assets/wt2_test_ids.npy")[5000:5256]; cap={}
g.forward(ids, hook=lambda n,X: cap.__setitem__("X",X.astype(np.float64)) if n=="h6.mlp.c_fc" else None, upto_layer=7)
X=cap["X"]; W=g.P(6,"mlp.c_fc.weight").astype(np.float64)
rho=(X**2).mean(0); Y=X@W
gamma=float((rho@(W**2)).sum()/(Y**2).mean(0).sum())
print("gamma",gamma)
json.dump({"gamma_w":gamma,"measured_ratio_large_beta_bf16":0.35},open('/home/claude/code/results/gamma_gpt2.json','w'))
