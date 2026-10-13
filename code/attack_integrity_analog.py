"""Byzantine steering against the ANALOG code with a residual (parity) test.

Test statistic per coded GEMM: normalized residual  ||(I - E E^+) Z|| / ||Z||  (scale-free).
Threshold: 1.25 x the largest honest value over calibration prompts (per layer).
Three runs per (format, beta): clean (false alarms), steer (same steering strength that takes over
GPT-2 under naive FF decoding), stealth (largest perturbation that keeps the statistic under the
threshold, assuming the attacker knows it).
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from gpt2np import GPT2, Tokenizer
from coded import AnalogCode
from attack_integrity import get_prompts, metrics, TARGET_OPS, EVIL, N, K, T

OUT = "/home/claude/code/results"


def run_cfg(g, calib, prompts, target, fmt, beta, E, gamma=30.0):
    v = g.wte[target].astype(np.float64); v /= np.linalg.norm(v)
    Pp = np.eye(N) - E @ np.linalg.pinv(E)
    D = np.linalg.pinv(E)[:K]
    pe = np.linalg.norm(Pp[:, EVIL])

    def stat(Z):
        Zf = Z.reshape(N, -1)
        return float(np.linalg.norm(Pp @ Zf) / (np.linalg.norm(Zf) + 1e-30))

    def make(mode, thr):
        c = AnalogCode(N, K, T, fmt=fmt, beta=beta, E=E, seed=17)
        c.stats = []
        def evil(name, Z, ctx):
            if mode != "clean" and name in TARGET_OPS:
                if mode == "steer":
                    cval = gamma / np.abs(D[:, EVIL]).max()
                else:   # stealth: residual added = 0.5 * thr * ||Z||
                    cval = 0.5 * thr[name] * np.linalg.norm(Z) * np.sqrt(N) / (pe * np.sqrt(Z.shape[0]))
                Z = Z + cval * v[None, :]
            return Z
        # wrap: compute stat after all responses (attack applied in byzantine hook)
        orig = c.linear
        def lin(name, X, W, b_):
            c._last = name
            return orig(name, X, W, b_)
        c.byzantine = {EVIL: evil}
        return c, lin

    def run(mode, thr, ps):
        c, lin = make(mode, thr)
        hits, nlls, flags = [], [], []
        for ids in ps:
            c.log.clear()
            logits = g.forward(ids, linear=lin)
            nll, hit = metrics(logits, ids, target)
            nlls.append(nll); hits.append(hit)
            st = {r["name"]: r["resid"] / (r["znorm"] + 1e-30) for r in c.log}
            flags.append(st)
        return hits, nlls, flags

    _, _, cal = run("clean", None, calib)
    thr = {n: 1.25 * max(f[n] for f in cal) for n in TARGET_OPS}
    res = {}
    for mode in ["clean", "steer", "stealth"]:
        hits, nlls, flags = run(mode, thr, prompts)
        det = [any(f[n] > thr[n] for n in TARGET_OPS) for f in flags]
        res[mode] = {"target_hit": float(np.mean(hits)), "ppl": float(np.exp(np.mean(nlls))), "flagged_frac": float(np.mean(det))}
    res["thr"] = thr
    print(fmt, beta, res, flush=True)
    return res


def main():
    tok, g = Tokenizer(), GPT2()
    target = tok.encode(" Paris")[0]
    calib, prompts = get_prompts(6, 5), get_prompts(6, 6)
    E = np.load("/home/claude/code/realcode_8_4_2.npy")
    out = {}
    for fmt, beta in [("bf16", 1.0), ("bf16", 10.0), ("fp32", 1000.0), ("fp32", 10000.0)]:
        out[f"{fmt}_beta{beta:g}"] = run_cfg(g, calib, prompts, target, fmt, beta, E)
        json.dump(out, open(os.path.join(OUT, "attack_integrity_analog.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
