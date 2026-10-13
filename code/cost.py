"""Analytic accounting: (i) fraction of prefill FLOPs that are public-weight x private-activation
GEMMs (offloadable), (ii) RNS residue count, (iii) per-worker latency and total work relative to a
local BF16 GEMM, from dense H100-SXM peak throughputs (NVIDIA datasheet)."""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coded import primes_for

PEAK = {"int8": 1979e12, "bf16": 989e12, "fp16": 989e12, "fp32": 67e12}
MODELS = {  # d, kv_dim, d_ff, layers, gated
    "GPT-2 (124M)": (768, 768, 3072, 12, False),
    "Llama-3-8B": (4096, 1024, 14336, 32, True),
    "Qwen2.5-7B": (3584, 512, 18944, 28, True),
    "Llama-3-70B": (8192, 1024, 28672, 80, True),
}


def linear_flops_per_token(d, kv, ff, gated):
    attn_proj = d * (d + 2 * kv) + d * d
    mlp = (3 if gated else 2) * d * ff
    return 2 * (attn_proj + mlp)


def offload_fraction(model, m):
    d, kv, ff, L, gated = MODELS[model]
    lin = linear_flops_per_token(d, kv, ff, gated) * m
    att = 2 * m * m * d                    # causal QK^T + AV
    return lin / (lin + att)


def residues(din, abits=8, wbits=8):
    return len(primes_for((2 ** (abits - 1) - 1) * (2 ** (wbits - 1) - 1) * din))


def worker_ratio(scheme, k, r=4):
    """Per-worker GEMM time / local BF16 GEMM time (same layer, compute-bound)."""
    if scheme == "ff":
        return r * PEAK["bf16"] / (PEAK["int8"] * k)
    if scheme == "analog_fp32":
        return PEAK["bf16"] / (PEAK["fp32"] * k)
    if scheme == "analog_bf16":
        return 1.0 / k
    if scheme == "replicate":
        return 1.0
    raise ValueError


if __name__ == "__main__":
    out = {"offload": {}, "residues": {}, "worker": {}}
    for mname in MODELS:
        out["offload"][mname] = {m: round(100 * offload_fraction(mname, m), 1) for m in (512, 2048, 8192, 32768)}
        d, kv, ff, L, g = MODELS[mname]
        out["residues"][mname] = {"A8 (d_model)": residues(d), "A8 (d_ff)": residues(ff),
                                  "A12 (d_ff)": residues(ff, 12), "A16 (d_ff)": residues(ff, 16)}
    N, k, T = 8, 4, 2
    for s in ["ff", "analog_fp32", "analog_bf16"]:
        out["worker"][s] = {"per_worker": round(worker_ratio(s, k), 3), "total_work": round(N * worker_ratio(s, k), 2)}
    for k_ in (2, 4, 6):
        out["worker"][f"ff_k{k_}_r4"] = round(worker_ratio("ff", k_, 4), 3)
        out["worker"][f"ff_k{k_}_r5"] = round(worker_ratio("ff", k_, 5), 3)
    print(json.dumps(out, indent=1))
    json.dump(out, open("/home/claude/code/results/cost.json", "w"), indent=1)
