"""Byzantine steering by one malicious worker (out of N=8) against the finite-field (RNS) code.

The malicious worker tampers with its returned residues in the MLP down-projections of the last
four blocks, adding (through the public decoder coefficients) an integer offset that steers decoded
block 0 toward a target token.
  naive : coordinator MDS-decodes the first k+T responses without checking
  rs    : coordinator runs the exact Reed-Solomon consistency check, locates and excludes the worker
The analog-code counterpart (residual test) is in attack_integrity_analog.py.  The FF results used in
the paper are stored in results/attack_integrity_ff.json.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from gpt2np import GPT2, Tokenizer, log_softmax
from coded import FFCode, AnalogCode, mat_inv_mod, quant_act

A = "/home/claude/assets"; OUT = "/home/claude/code/results"
N, K, T, M = 8, 4, 2, 256
EVIL = 3
TARGET_OPS = [f"h{l}.mlp.c_proj" for l in (8, 9, 10, 11)]


def get_prompts(n, seed):
    ids = np.load(os.path.join(A, "wt2_test_ids.npy"))
    rng = np.random.default_rng(seed)
    return [ids[s:s + M] for s in rng.choice(len(ids) - M, n, replace=False)]


def metrics(logits, ids, target, positions=None):
    lp = log_softmax(logits.astype(np.float64))
    nll = -lp[np.arange(len(ids) - 1), ids[1:]]
    nll = np.nan_to_num(nll, nan=50.0, posinf=50.0)
    pos = np.arange(len(ids)) if positions is None else positions
    hit = float((logits[pos].argmax(-1) == target).mean())
    return float(nll.mean()), hit


def ff_experiment(g, prompts, target, gamma):
    v = g.wte[target].astype(np.float64); v /= np.linalg.norm(v)
    b = M // K
    out = {}
    for mode in ["clean", "naive", "rs"]:
        c = FFCode(N, K, T, abits=8, seed=7)
        c.verify = (mode != "naive")
        if mode != "clean":
            def evil(name, pi, p, Z, ctx):
                if name not in TARGET_OPS:
                    return Z
                # inject an integer offset u into decoded block 0 (rows of block 0) using the public code
                use = list(range(K + T))                     # first k+T arrivals (predictable)
                Vinv = mat_inv_mod(ctx.vander(p, use), p)
                coef = Vinv[0, use.index(EVIL)]
                Wq, sw = ctx.qc.c[name]
                u = np.rint(gamma * v / (ctx._sx_med * sw)).astype(np.int64)  # attacker's scale guess
                delta = (u % p) * pow(int(coef), p - 2, p) % p
                return (Z + delta[None, :]) % p
            c.byzantine = {EVIL: evil}
        nlls, hits, det = [], [], []
        for ids in prompts:
            c._sx_med = 0.026
            logits = g.forward(ids, linear=c.linear)
            nll, hit = metrics(logits, ids, target, positions=np.arange(b))   # block-0 positions
            nlls.append(nll); hits.append(hit)
            det.append(any(r["detected"] for r in c.log if r["name"] in TARGET_OPS)); c.log.clear()
        out[mode] = {"nll": float(np.mean(nlls)), "ppl": float(np.exp(np.mean(nlls))), "target_hit_block0": float(np.mean(hits)),
                     "detected_frac": float(np.mean(det))}
        print("FF", mode, out[mode], flush=True)
    return out


def analog_experiment(g, calib, prompts, target, fmt, beta, E):
    v = g.wte[target].astype(np.float64); v /= np.linalg.norm(v)
    # 1) calibrate per-op detection threshold on honest runs
    thr = {}
    c = AnalogCode(N, K, T, fmt=fmt, beta=beta, E=E, seed=11)
    for ids in calib:
        g.forward(ids, linear=c.linear)
        for r in c.log:
            thr[r["name"]] = max(thr.get(r["name"], 0.0), r["resid"])
        c.log.clear()
    Pperp = np.eye(N) - E @ np.linalg.pinv(E)
    pe = np.linalg.norm(Pperp[:, EVIL])
    D = np.linalg.pinv(E)[:K]
    b = M // K
    res = {}
    for mode in ["clean", "stealth"]:
        c = AnalogCode(N, K, T, fmt=fmt, beta=beta, E=E, seed=13)
        if mode == "stealth":
            def evil(name, Z, ctx):
                if name not in TARGET_OPS:
                    return Z
                a = 0.5 * thr[name]                       # residual budget the attacker spends
                cval = a / (pe * np.sqrt(Z.shape[0]))     # per-row amplitude along v
                return Z + cval * v[None, :]
            c.byzantine = {EVIL: evil}
        nlls, hits, det, rel = [], [], [], []
        for ids in prompts:
            logits = g.forward(ids, linear=c.linear)
            nll, hit = metrics(logits, ids, target)
            nlls.append(nll); hits.append(hit)
            det.append(any(r["resid"] > thr[r["name"]] for r in c.log if r["name"] in TARGET_OPS)); c.log.clear()
        res[mode] = {"ppl": float(np.exp(np.mean(nlls))), "target_hit": float(np.mean(hits)), "detected_frac": float(np.mean(det))}
    # decoded perturbation size per row, relative to the typical honest output row norm of that op
    res["stealth_amp_per_row"] = {n: float(0.5 * thr[n] / (pe * np.sqrt(b)) * np.abs(D[:, EVIL]).max()) for n in TARGET_OPS}
    print("analog", fmt, beta, res, flush=True)
    return res


def main():
    tok, g = Tokenizer(), GPT2()
    target = tok.encode(" Paris")[0]
    calib, prompts = get_prompts(3, 5), get_prompts(6, 6)
    E = np.load("/home/claude/code/realcode_8_4_2.npy")
    results = {"target": " Paris", "evil_worker": EVIL, "ops": TARGET_OPS, "N": N, "k": K, "T": T, "m": M}
    # honest output-row norms of the target ops (for normalizing the stealth amplitude)
    norms = {n: [] for n in TARGET_OPS}
    def hook_out(name, X, W, b):
        Y = X @ W + b
        if name in norms: norms[name].append(float(np.median(np.linalg.norm(Y, axis=1))))
        return Y
    for ids in prompts[:2]:
        g.forward(ids, linear=hook_out)
    results["honest_row_norm"] = {n: float(np.mean(v)) for n, v in norms.items()}
    results["ff"] = ff_experiment(g, prompts, target, gamma=float(sys.argv[1]) if len(sys.argv) > 1 else 30.0)
    results["analog"] = {}
    for fmt in ["bf16", "fp32"]:
        for beta in [1.0, 10.0, 100.0, 1000.0]:
            results["analog"][f"{fmt}_beta{beta:g}"] = analog_experiment(g, calib, prompts, target, fmt, beta, E)
            json.dump(results, open(os.path.join(OUT, "attack_integrity.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
