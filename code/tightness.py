"""Theorem 1(b): the barrier is tight. Code with k=1, T masks, c_i=1, mask rows b_i (equal norm, sum 0,
any T rows independent) and the averaging decoder lambda_i = 1/N achieves relMSE = e^{2 eps} x bound
under share-storage rounding. Checked on a real GPT-2 GEMM (layer-6 FFN up-projection)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from gpt2np import GPT2
from coded import to_fmt
U = {"bf16": 2.0 ** -8, "fp16": 2.0 ** -11, "fp32": 2.0 ** -24}
g = GPT2()
ids = np.load("/home/claude/assets/wt2_test_ids.npy")[5000:5256]
cap = {}
g.forward(ids, hook=lambda n, X: cap.__setitem__("X", X.astype(np.float64)) if n == "h6.mlp.c_fc" else None, upto_layer=7)
W = g.P(6, "mlp.c_fc.weight").astype(np.float64)
rng = np.random.default_rng(0)
DATA = sys.argv[1] if len(sys.argv) > 1 else "gaussian"
X = cap["X"] if DATA == "gpt2" else rng.standard_normal(cap["X"].shape)
Y = X @ W
N, T = 9, 2
ang = 2 * np.pi * np.arange(N) / N
Bm = np.stack([np.cos(ang), np.sin(ang)], 1)          # sum 0, unit norm, no two rows parallel
rms = np.sqrt((X ** 2).mean(0))
out = []
for fmt in ["bf16", "fp16", "fp32"]:
    s = rng.standard_normal(2_000_000) * 3
    kappa = float(((to_fmt(s, fmt).astype(np.float64) - s) ** 2).mean() / (U[fmt] ** 2 * (s ** 2).mean()))
    for beta in [1, 3, 10, 100, 1000, 1e4]:
        R = rng.standard_normal((T,) + X.shape) * (beta * rms)
        Z = np.zeros_like(Y)
        for i in range(N):
            S = X + Bm[i, 0] * R[0] + Bm[i, 1] * R[1]
            with np.errstate(over="ignore"):
                Sq = to_fmt(S, fmt).astype(np.float64)     # storage rounding only (Assumption 1)
            Z += Sq @ W / N                               # exact GEMM + exact averaging decoder
        rel = float(((Z - Y) ** 2).mean() / (Y ** 2).mean()) if np.all(np.isfinite(Z)) else float("inf")
        eps = 0.5 * np.log1p(1.0 / beta ** 2)              # per-worker per-entry leakage (unit-norm b_i)
        bound = kappa * U[fmt] ** 2 / (N * np.expm1(2 * eps))
        out.append({"fmt": fmt, "beta": beta, "rel_mse": rel, "bound": bound, "ratio": rel / bound, "e2eps": float(np.exp(2 * eps))})
        print(fmt, beta, f"{rel:.3e} {bound:.3e} ratio={rel/bound:.3f} e^2eps={np.exp(2*eps):.3f}", flush=True)
json.dump(out, open(f"/home/claude/code/results/tightness_{DATA}.json", "w"), indent=1)
