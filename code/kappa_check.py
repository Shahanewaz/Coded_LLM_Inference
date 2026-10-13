"""Check of Assumption 1: kappa = E[e^2] / (u^2 E[s^2]) for storage rounding of analog shares
(data + Gaussian mask) built from real GPT-2 activations (layer-6 FFN input), per format and mask ratio.
Also reports the fraction of share entries that overflow or fall in the subnormal range."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from gpt2np import GPT2
from coded import to_fmt
U = {"bf16": 2.0 ** -8, "fp16": 2.0 ** -11, "fp32": 2.0 ** -24}
TINY = {"bf16": 1.18e-38, "fp16": 6.1e-5, "fp32": 1.18e-38}
MAXV = {"bf16": 3.39e38, "fp16": 65504.0, "fp32": 3.4e38}
g = GPT2()
ids = np.load("/home/claude/assets/wt2_test_ids.npy")[5000:5256]
cap = {}
g.forward(ids, hook=lambda n, X: cap.__setitem__("X", X.astype(np.float64)) if n == "h6.mlp.c_fc" else None, upto_layer=7)
X = cap["X"]; rms = np.sqrt((X ** 2).mean(0))
rng = np.random.default_rng(0)
out = []
for src in ["gaussian", "gpt2"]:
    D = rng.standard_normal(X.shape) * rms if src == "gaussian" else X
    for beta in [0, 1, 10, 100, 1000]:
        S = D + beta * rms * rng.standard_normal(X.shape)
        for fmt in ["bf16", "fp16", "fp32"]:
            with np.errstate(over="ignore", invalid="ignore"):
                Sq = to_fmt(S, fmt).astype(np.float64)
            fin = np.isfinite(Sq) & (np.abs(S) < MAXV[fmt])
            e = (Sq - S)[fin]
            kappa = float((e ** 2).mean() / (U[fmt] ** 2 * (S[fin] ** 2).mean()))
            sub = float(((np.abs(S) < TINY[fmt]) & (S != 0)).mean())
            out.append({"data": src, "beta": beta, "fmt": fmt, "kappa": kappa, "overflow_frac": float(1 - fin.mean()), "subnormal_frac": sub})
            print(out[-1], flush=True)
json.dump(out, open("/home/claude/code/results/kappa_check.json", "w"), indent=1)
