"""
Diagnostics for two observations on the RTX 5000 Ada (run AFTER run_main.bat has finished):

  int8   INT8 vs BF16 GEMM throughput on this GPU: torch._int_mm alone (no mod, no int64 conversion),
         two operand layouts, plus the cost of the extra steps a coded worker performs.
  relerr Theorem 1 at the level it is stated: relative MSE of analog-coded projections on REAL
         activations of Qwen2.5-7B and Qwen2.5-1.5B (exact float64 reference), per format and beta,
         next to the bound kappa*u^2/sum_i(e^{2 eps_i}-1) (with and without the correlation factor gamma).
  noise  Controls for "analog BF16 at beta=1/10 has lower perplexity than BF16" on Qwen2.5-7B:
         several mask seeds, an FP32-exact linear path, and FP32 + matched Gaussian output noise.

Usage (Anaconda Prompt, env cpd, in Documents\\cpd_gpu):
  python cpd_diag.py                 # all three, ~30-45 min
  python cpd_diag.py --stages int8   # just one
Writes results_diag_<stage>.json.
"""
import argparse, gc, json, math, os, sys, time, traceback
from types import SimpleNamespace
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpd_ada as D
G, A = D.G, D.A
DEV = G.DEV
G.LOGFILE = "cpd_diag.log"
log = G.log
D.ARGS = SimpleNamespace(drop_bf16=[])
U = {"bf16": 2.0 ** -8, "fp16": 2.0 ** -11, "fp32": 2.0 ** -24}
KAPPA = 0.18


def bench(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


# ------------------------------------------------------------------ 1. INT8 throughput
@torch.no_grad()
def stage_int8(args):
    out = []
    shapes = {"Llama-3-8B gate/up": (4096, 14336), "Llama-3-8B down": (14336, 4096), "Qwen2.5-7B up": (3584, 18944)}
    ms = [1024, 4096] if args.quick else [512, 1024, 2048, 4096, 8192]
    for m in ms:
        for nm, (d, n) in shapes.items():
            rec = {"shape": nm, "m": m, "d": d, "n": n}
            try:
                flops = 2 * m * d * n
                A16 = torch.randn(m, d, device=DEV, dtype=torch.bfloat16)
                B16 = torch.randn(d, n, device=DEV, dtype=torch.bfloat16)
                rec["bf16_TFLOPS"] = flops / bench(lambda: A16 @ B16) / 1e12
                H16, K16 = A16.half(), B16.half()
                rec["fp16_TFLOPS"] = flops / bench(lambda: H16 @ K16) / 1e12
                del H16, K16
                A8 = torch.randint(-125, 126, (m, d), device=DEV, dtype=torch.int8)
                Bcm = torch.randint(-127, 128, (n, d), device=DEV, dtype=torch.int8).t()          # column-major [d, n]
                rec["int8_colmajorB_TOPS"] = flops / bench(lambda: torch._int_mm(A8, Bcm)) / 1e12
                try:
                    Brm = Bcm.contiguous()                                                       # row-major [d, n]
                    rec["int8_rowmajorB_TOPS"] = flops / bench(lambda: torch._int_mm(A8, Brm)) / 1e12
                    del Brm
                except Exception as e:
                    rec["int8_rowmajorB_TOPS"] = "unsupported: " + repr(e)[:80]
                C = torch._int_mm(A8, Bcm)
                t_gemm = bench(lambda: torch._int_mm(A8, Bcm))
                rec["mod_p_ms"] = bench(lambda: torch.remainder(C, 251)) * 1e3
                rec["to_int64_ms"] = bench(lambda: C.to(torch.int64)) * 1e3
                rec["int8_gemm_ms"] = t_gemm * 1e3
                rec["bf16_gemm_ms"] = flops / rec["bf16_TFLOPS"] / 1e12 * 1e3
                Xb = A16
                rec["quant_rows_ms"] = bench(lambda: G.quant_rows(Xb, 8)) * 1e3
                rec["int8_over_bf16_speed"] = rec["int8_colmajorB_TOPS"] / rec["bf16_TFLOPS"]
                del A16, B16, A8, Bcm, C
            except Exception as e:
                rec["error"] = repr(e)[:300]
            torch.cuda.empty_cache()
            out.append(rec)
            log("int8", {a: (round(b, 3) if isinstance(b, float) else b) for a, b in rec.items()})
    return {"device": torch.cuda.get_device_name(0), "records": out}


# ------------------------------------------------------------------ 2. relative error vs Theorem 1
def bound_terms(E, k, T, beta):
    """sum_i (e^{2 eps_i} - 1) with eps_i = 1/2 log(1 + ||c_i||^2 / (beta^2 ||b_i||^2))."""
    c = E[:, :k].double(); b = E[:, k:].double()
    return float(((c.norm(dim=1) ** 2) / (beta ** 2 * b.norm(dim=1) ** 2)).sum())


@torch.no_grad()
def stage_relerr(args):
    res = {}
    E = A.make_real_code(iters=60 if args.quick else 600, restarts=1 if args.quick else 3)
    k, T = 4, 2
    for name in args.relerr_models:
        try:
            tok, model, state = D.load_model(name)
            wins = G.wikitext_windows(tok, 1 if args.quick else 2, 256 if args.quick else 512)
            nL = len(model.model.layers)
            targets = {"L0.q_proj": model.model.layers[0].self_attn.q_proj,
                       f"L{nL//2}.mlp.up_proj": model.model.layers[nL // 2].mlp.up_proj,
                       f"L{nL//2}.mlp.down_proj": model.model.layers[nL // 2].mlp.down_proj,
                       f"L{nL-1}.mlp.down_proj": model.model.layers[nL - 1].mlp.down_proj}
            caps = {}
            hooks = [mod.register_forward_pre_hook(lambda m_, inp, key=key: caps.setdefault(key, []).append(inp[0][0].detach()))
                     for key, mod in targets.items()]
            for w in wins:
                model(w.unsqueeze(0).to(DEV))
            for h in hooks:
                h.remove()
            r = res.setdefault(name, {})
            for key, mod in targets.items():
                X = torch.cat(caps[key], 0)                                  # [m, d] bf16
                W = mod.lin.weight.t()                                       # [d, n] bf16
                ncol = min(W.shape[1], 2048)
                W = W[:, :ncol]
                Y = X.double() @ W.double()
                ey = (Y ** 2).mean().item()
                rho = (X.double() ** 2).mean(0)                              # per-channel second moment
                gamma = float((rho @ (W.double() ** 2)).sum() / (Y ** 2).mean(0).sum())
                rr = {"m": X.shape[0], "d": X.shape[1], "cols": ncol, "gamma": gamma}
                rr["plain_bf16"] = float(((X @ W).double() - Y).pow(2).mean().item() / ey)
                for fmt in ["bf16", "fp16", "fp32"]:
                    for beta in [1.0, 10.0, 100.0, 1000.0]:
                        code = A.AnalogCode(E, k, T, fmt, beta, seed=3)
                        Yh = code.linear(X.float(), W, None).double()          # FP32 output: no final BF16 rounding
                        rel = float((Yh - Y).pow(2).mean().item() / ey) if torch.isfinite(Yh).all() else float("inf")
                        B = KAPPA * U[fmt] ** 2 / bound_terms(E, k, T, beta)
                        rr[f"{fmt}_beta{beta:g}"] = {"rel_mse": rel, "bound": B, "gamma_bound": gamma * B,
                                                      "ratio_to_gamma_bound": rel / (gamma * B) if rel != float("inf") else None}
                r[key] = rr
                log(name, key, {a: (round(b["rel_mse"], 8) if isinstance(b, dict) else b) for a, b in rr.items()})
                G.jdump(res, "results_diag_relerr.partial.json")
            del model, state, caps; gc.collect(); torch.cuda.empty_cache()
        except Exception:
            res.setdefault(name, {})["error"] = traceback.format_exc()[-1200:]
            log("ERROR relerr", name, traceback.format_exc()[-500:]); gc.collect(); torch.cuda.empty_cache()
    return res


# ------------------------------------------------------------------ 3. noise controls
_orig_forward = D.LazyCodedLinear.forward


def _forward(self, x):
    mode = self.state["mode"]
    if mode not in ("fp32", "fp32noise"):
        return _orig_forward(self, x)
    shp = x.shape
    X = x.reshape(-1, shp[-1]).float()
    Y = X @ self.lin.weight.float().t()
    if self.lin.bias is not None:
        Y = Y + self.lin.bias.float()
    if mode == "fp32noise":
        r = self.state["noise_rel"]
        g = self.state["gen"]
        Y = Y + math.sqrt(r) * Y.pow(2).mean().sqrt() * torch.randn(Y.shape, generator=g, device=Y.device)
    return Y.to(x.dtype).reshape(*shp[:-1], -1)


D.LazyCodedLinear.forward = _forward


@torch.no_grad()
def stage_noise(args):
    name = args.noise_model
    res = {"model": name}
    tok, model, state = D.load_model(name)
    wins = G.wikitext_windows(tok, 2 if args.quick else 20, 256 if args.quick else 1024)
    asub = wins[:max(1, len(wins) // 2)]                        # same subset as the quality stage
    E = A.make_real_code(iters=60 if args.quick else 600, restarts=1 if args.quick else 3)

    def run(tag, mode, code=None, **extra):
        state.clear(); state.update({"mode": mode, "code": code, "abits": 8, "attn": "plain", **extra})
        t = time.time()
        p, cnt, bad = D.ppl(model, asub)
        res[tag] = {"ppl": p, "tokens": cnt, "nonfinite_windows": bad, "sec": time.time() - t}
        log(name, tag, res[tag]); G.jdump(res, "results_diag_noise.partial.json")

    run("bf16", "plain")
    run("fp32_linear", "fp32")
    for beta in [1.0, 10.0]:
        for seed in [3, 4, 5]:
            run(f"analog_bf16_beta{beta:g}_seed{seed}", "analog", code=A.AnalogCode(E, 4, 2, "bf16", beta, seed=seed))
    for r in [1e-5, 1e-4, 1e-3, 1e-2]:
        for seed in [0, 1]:
            g = torch.Generator(device=DEV); g.manual_seed(seed)
            run(f"fp32_plus_noise_rel{r:g}_seed{seed}", "fp32noise", noise_rel=r, gen=g)
    return res


# ------------------------------------------------------------------ 4. GSM8K harness check
@torch.no_grad()
def stage_gsm8k(args):
    """Same prompts and greedy decoding as cpd_gpu2.stage_generate, but saves the generated texts, uses a longer
    budget, adds an FP32-linear reference and a lenient numeric scorer, to explain BF16-vs-W8A8 gaps."""
    import re
    ds = G.ds_load("openai/gsm8k", "main", split="test")
    train = G.ds_load("openai/gsm8k", "main", split="train")
    shots = "".join(f"Question: {train[i]['question']}\nAnswer: {train[i]['answer']}\n\n" for i in range(4))
    n = 3 if args.quick else args.gsm_n
    probs = [ds[i] for i in range(n)]

    def extract(t):
        m = re.findall(r"####\s*(-?[\d,\.]+)", t)
        if not m:
            m = re.findall(r"(-?\d[\d,]*\.?\d*)", t)
        return m[-1].replace(",", "").rstrip(".") if m else None

    def num_eq(a, b):
        try:
            return a is not None and b is not None and abs(float(a) - float(b)) < 1e-6
        except ValueError:
            return False

    tok, model, state = D.load_model(args.gsm_model)

    def greedy(prompt, max_new):
        ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
        out = model(ids, use_cache=True)
        past = out.past_key_values; nxt = out.logits[0, -1].argmax().view(1, 1); gen = [nxt.item()]
        stop = "budget"
        for _ in range(max_new - 1):
            o = model(nxt, past_key_values=past, use_cache=True)
            past = o.past_key_values; nxt = o.logits[0, -1].argmax().view(1, 1); gen.append(nxt.item())
            txt = tok.decode(gen)
            if "\n\nQuestion" in txt or "Question:" in txt[-12:]:
                stop = "question"; break
            if nxt.item() == tok.eos_token_id:
                stop = "eos"; break
        return gen, stop

    res = {"model": args.gsm_model, "n": n, "max_new": args.gsm_max}
    for tag, mode in [("bf16", "plain"), ("fp32_linear", "fp32"), ("w8a8", "wq")]:
        state.clear(); state.update({"mode": mode, "code": None, "abits": 8, "attn": "plain"})
        t = time.time(); recs = []
        for ex in probs:
            g, stop = greedy(shots + f"Question: {ex['question']}\nAnswer:", args.gsm_max)
            txt = tok.decode(g); gold = extract(ex["answer"]); pred = extract(txt)
            recs.append({"stop": stop, "tokens": len(g), "has_hashes": "####" in txt, "pred": pred, "gold": gold,
                         "strict": pred == gold, "lenient": num_eq(pred, gold), "text": txt})
        res[tag] = {"strict_acc": sum(r["strict"] for r in recs) / n, "lenient_acc": sum(r["lenient"] for r in recs) / n,
                    "hit_budget": sum(r["stop"] == "budget" for r in recs), "no_hashes": sum(not r["has_hashes"] for r in recs),
                    "sec": time.time() - t, "records": recs}
        log(args.gsm_model, tag, {a: b for a, b in res[tag].items() if a != "records"})
        G.jdump(res, "results_diag_gsm8k.partial.json")
    for a, b in [("bf16", "w8a8"), ("bf16", "fp32_linear"), ("fp32_linear", "w8a8")]:
        res[f"same_text_{a}_vs_{b}"] = sum(x["text"] == y["text"] for x, y in zip(res[a]["records"], res[b]["records"]))
    log("identical generations", {k: v for k, v in res.items() if k.startswith("same_text")})
    return res


STAGES = {"int8": stage_int8, "relerr": stage_relerr, "noise": stage_noise, "gsm8k": stage_gsm8k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["int8", "relerr", "noise", "gsm8k"])
    ap.add_argument("--gsm-model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--gsm-n", type=int, default=40)
    ap.add_argument("--gsm-max", type=int, default=400)
    ap.add_argument("--relerr-models", nargs="+", default=["Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-1.5B"])
    ap.add_argument("--noise-model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.relerr_models = ["Qwen/Qwen2.5-0.5B"]; args.noise_model = "Qwen/Qwen2.5-0.5B"; args.gsm_model = "Qwen/Qwen2.5-0.5B"; args.gsm_max = 16
    G.QUICK = args.quick
    pre = "quick_" if args.quick else ""
    env = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
           "tf32": torch.backends.cuda.matmul.allow_tf32, "args": vars(args)}
    log("environment", env)
    for st in args.stages:
        path = f"{pre}results_diag_{st}.json"
        if os.path.exists(path) and not args.force:
            log(path, "exists, skipping"); continue
        log(f"===== diag: {st} =====")
        t = time.time()
        try:
            G.jdump({"result": STAGES[st](args), "env": env, "seconds": time.time() - t}, path)
            log(f"diag {st} finished in {time.time()-t:.0f}s -> {path}")
        except Exception:
            log(f"diag {st} FAILED", traceback.format_exc()[-1500:])
            G.jdump({"error": traceback.format_exc()[-3000:], "env": env}, f"{pre}results_diag_{st}.FAILED.json")
        gc.collect(); torch.cuda.empty_cache()
    log("DIAG ALL DONE")


if __name__ == "__main__":
    main()
