"""Empirical check of the precision-privacy barrier (Theorem 1) on a real GPT-2 GEMM.

For the MLP up-projection of block 6 on a WikiText-2 prompt (m=256), run the analog code
(N=8, k=4, T=2, optimized real code) in BF16 / FP16 / FP32 for a sweep of mask ratios beta and
compare the measured relative MSE of the decoded product with the lower bound
    kappa_f u_f^2 / sum_i (exp(2 eps_i) - 1),
where eps_i is worker i's per-entry Gaussian leakage and kappa_f is the measured mean-square
relative rounding error of the format (in units of u_f^2) on Gaussian data.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from gpt2np import GPT2
from coded import AnalogCode, to_fmt, FFCode, int_reference_linear

U = {"bf16": 2.0 ** -8, "fp16": 2.0 ** -11, "fp32": 2.0 ** -24}


def kappa(fmt, rng):
    s = rng.standard_normal(2_000_000) * 3.0
    e = to_fmt(s, fmt).astype(np.float64) - s
    return float((e ** 2).mean() / (U[fmt] ** 2 * (s ** 2).mean()))


def main():
    rng = np.random.default_rng(0)
    g = GPT2()
    ids = np.load("/home/claude/assets/wt2_test_ids.npy")[5000:5256]
    cap = {}
    def hook(name, X):
        if name == "h6.mlp.c_fc": cap["X"] = X.astype(np.float64)
    g.forward(ids, hook=hook, upto_layer=7)
    X = cap["X"]; W = g.P(6, "mlp.c_fc.weight").astype(np.float64); b = np.zeros(W.shape[1])
    Y = X @ W
    E = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "realcode_8_4_2.npy"))
    k, T = 4, 2
    C, B = E[:, :k], E[:, k:]
    out = {"kappa": {f: kappa(f, rng) for f in U}, "rows": []}
    print(out["kappa"])
    sig = float((Y ** 2).mean())
    for beta in np.logspace(0, 5, 11):
        # per-worker per-entry Gaussian leakage (x ~ N(0, rho), mask sd = beta * sqrt(rho))
        eps_i = 0.5 * np.log1p((C ** 2).sum(1) / (beta ** 2 * (B ** 2).sum(1)))
        denom = float(np.expm1(2 * eps_i).sum())
        row = {"beta": float(beta), "eps_worker_max": float(eps_i.max())}
        for fmt in U:
            c = AnalogCode(8, k, T, fmt=fmt, beta=float(beta), E=E, seed=3)
            with np.errstate(all="ignore"):
                Yh = c.linear("x", X.astype(np.float32), W.astype(np.float32), b).astype(np.float64)
            rel = float(((Yh - Y) ** 2).mean() / sig) if np.all(np.isfinite(Yh)) else float("inf")
            row[fmt] = rel
            row[fmt + "_bound"] = out["kappa"][fmt] * U[fmt] ** 2 / denom
        out["rows"].append(row)
        print(row, flush=True)
    # finite-field reference: exact relative to the W8A8 product
    ff = FFCode(8, k, T); ff.check_exact = True
    Yff = ff.linear("x", X.astype(np.float32), W.astype(np.float32), b).astype(np.float64)
    out["ff_exact_vs_w8a8"] = ff.log[-1]["exact"]
    out["w8a8_rel_mse"] = float(((Yff - Y) ** 2).mean() / sig)
    out["bf16_plain_rel_mse"] = float(((to_fmt(to_fmt(X, "bf16").astype(np.float32) @ to_fmt(W, "bf16").astype(np.float32), "bf16") - Y) ** 2).mean() / sig)
    print(out["ff_exact_vs_w8a8"], out["w8a8_rel_mse"], out["bf16_plain_rel_mse"])
    json.dump(out, open("/home/claude/code/results/barrier.json", "w"), indent=1)


if __name__ == "__main__":
    main()
