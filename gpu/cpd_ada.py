"""
Larger-model experiments for "Quantize the Secret, Not the Share" on a 32 GB GPU (RTX 5000 Ada).
Reuses cpd_gpu2.py (RNS code, stages) and cpd_gpu.py (analog baseline) and adds:
  * models are loaded straight onto the GPU (no CPU copy),
  * INT8 weights are built lazily; for models too large for BF16 + INT8 together (Qwen2.5-14B),
    the BF16 weights are dropped after the BF16 baseline has been measured ("--drop-bf16"),
  * a 'quality' stage: WikiText-2 perplexity and LAMBADA for BF16 / W8A8 / W8A16 / RNS-coded
    (bit-exactness checked on every coded GEMM), a Byzantine worker (Reed-Solomon vs naive decoding),
    and optionally the analog baseline.

Every job writes results3_<tag>_<stage>.json and is skipped on re-run if that file exists.
  python cpd_ada.py --tag q7 --stages quality --models Qwen/Qwen2.5-7B --analog
  python cpd_ada.py --quick            # 5-minute self-test on two small models (Qwen + OLMo-2)
"""
import argparse, gc, json, math, os, sys, time, traceback, weakref
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpd_gpu2 as G
import cpd_gpu as A

DEV = G.DEV
G.LOGFILE = "cpd_ada.log"
log = G.log
REG = weakref.WeakSet()
DROP = {"on": False}


# ------------------------------------------------------------------ lazy INT8 layers
class LazyCodedLinear(nn.Module):
    def __init__(self, lin, state):
        super().__init__()
        self.lin, self.state = lin, state
        self.Wq = None
        self.sw = None
        REG.add(self)

    def materialize(self):
        if self.Wq is None:
            Wq, sw = G.quant_cols(self.lin.weight.data.t())
            self.Wq = Wq.to(torch.int8).t().contiguous().t()
            self.sw = sw
            del Wq
            if DROP["on"]:
                self.lin.weight = None

    def forward(self, x):
        mode = self.state["mode"]
        if mode == "plain":
            if self.lin.weight is None:
                raise RuntimeError("BF16 weights were dropped for this model; run BF16 first")
            return self.lin(x)
        self.materialize()
        shp = x.shape
        X = x.reshape(-1, shp[-1])
        b = self.lin.bias
        if mode == "wq":
            ab = self.state["abits"]
            Xq, sx = G.quant_rows(X, ab)
            Y = G.int_ref(Xq, self.Wq, ab).double() * sx.double() * self.sw.double()
            if b is not None:
                Y = Y + b.double()
            Y = Y.to(x.dtype)
        elif mode == "rns":
            Y = self.state["code"].linear(X, self.Wq, self.sw, b)
        elif mode == "analog":
            Y = self.state["code"].linear(X, self.lin.weight.t(), b)
        else:
            raise ValueError(mode)
        return Y.reshape(*shp[:-1], -1)


def materialize_all():
    for i, m in enumerate(list(REG)):
        m.materialize()
        if i % 16 == 0:
            torch.cuda.empty_cache()
    torch.cuda.empty_cache()


_orig_set_mode = G.set_mode


def set_mode(state, mode, code=None, abits=8):
    if mode not in ("plain", "analog"):
        materialize_all()
    _orig_set_mode(state, mode, code=code, abits=abits)


def load_model(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    gc.collect(); torch.cuda.empty_cache()
    DROP["on"] = name in ARGS.drop_bf16
    tok = AutoTokenizer.from_pretrained(name)
    kw = dict(attn_implementation="eager", device_map=DEV)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, **kw)
    model.eval()
    state = {"mode": "plain", "attn": "plain"}
    n = G.wrap_model(model, state)
    nparam = sum(p.numel() for p in model.parameters()) / 1e9
    log(f"loaded {name}: {nparam:.2f}B params, {n} linear layers wrapped, drop_bf16={DROP['on']}, "
        f"GPU mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    return tok, model, state


G.CodedLinear = LazyCodedLinear
G.set_mode = set_mode
G.load_model = load_model


# ------------------------------------------------------------------ quality stage
def load_lambada(tok, n):
    try:
        ds = G.ds_load("EleutherAI/lambada_openai", "default", split="test")
    except Exception:
        try:
            ds = G.ds_load("EleutherAI/lambada_openai", "en", split="test")
        except Exception as e:
            log("LAMBADA unavailable:", repr(e)[:150]); return []
    out = []
    for t in ds["text"][:n]:
        ctx, last = t.rsplit(" ", 1)
        c = tok(ctx).input_ids; w = tok(" " + last, add_special_tokens=False).input_ids
        out.append((torch.tensor(c + w), len(w)))
    return out


@torch.no_grad()
def hidden(model, ids):
    return model.model(ids).last_hidden_state[0]


@torch.no_grad()
def ppl(model, wins):
    nll, cnt, bad = 0.0, 0, 0
    for w in wins:
        ids = w.unsqueeze(0).to(DEV)
        h = hidden(model, ids)
        L = ids.shape[1]
        for s in range(0, L - 1, 256):
            e = min(s + 256, L - 1)
            lg = model.lm_head(h[s:e]).float()
            v = F.cross_entropy(lg, ids[0, s + 1:e + 1], reduction="none")
            if not torch.isfinite(v).all():
                bad += 1; v = torch.nan_to_num(v, nan=30.0, posinf=30.0)
            nll += v.sum().item(); cnt += v.numel()
    return math.exp(nll / cnt), cnt, bad


@torch.no_grad()
def lambada_acc(model, items):
    hit = 0
    for ids, nw in items:
        x = ids[:-1].unsqueeze(0).to(DEV)
        h = hidden(model, x)
        pred = model.lm_head(h[-nw:]).argmax(-1).cpu()
        hit += int(torch.equal(pred, ids[-nw:]))
    return hit / max(1, len(items))


def stage_quality(args):
    res = {}
    for name in args.models:
        try:
            tok, model, state = load_model(name)
            wins = G.wikitext_windows(tok, args.windows, args.seq)
            lam = load_lambada(tok, args.lambada) if args.lambada else []
            q = wins[:max(1, len(wins) // 4)]
            r = res.setdefault(name, {"windows": len(wins), "seq": args.seq, "lambada_n": len(lam),
                                      "drop_bf16": DROP["on"]})

            def run(tag, mode, abits=8, code=None, win=None, lambada=False):
                set_mode(state, mode, code=code, abits=abits)
                t = time.time()
                p, cnt, bad = ppl(model, wins if win is None else win)
                e = {"ppl": p, "tokens": cnt, "nonfinite_windows": bad}
                if lambada and lam:
                    e["lambada_acc"] = lambada_acc(model, lam)
                if isinstance(code, G.RNSCode):
                    e["stats"] = dict(code.stats)
                if isinstance(code, A.AnalogCode):
                    e["overflow"] = code.overflow
                e["sec"] = time.time() - t
                r[tag] = e
                log(name, tag, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in e.items()})
                jdump_partial(res)

            run("bf16", "plain", lambada=True)
            if args.analog and not DROP["on"]:
                E = A.make_real_code()
                asub = wins[:max(1, len(wins) // 2)]
                run("bf16_on_analog_subset", "plain", win=asub)
                for fmt, beta in [("bf16", 1.0), ("bf16", 10.0), ("bf16", 100.0), ("fp16", 100.0),
                                  ("fp16", 1000.0), ("fp32", 1e3), ("fp32", 1e4)]:
                    code = A.AnalogCode(E, 4, 2, fmt, beta, seed=3)
                    run(f"analog_{fmt}_beta{beta:g}", "analog", code=code, win=asub)
                    r[f"analog_{fmt}_beta{beta:g}"]["leak_per_entry_worst_pair_nats"] = A.worst_leak_per_entry(E, 4, 2, beta)
                run("w8a8_on_analog_subset", "wq", win=asub)
            run("w8a8", "wq", lambada=True)
            run("w8a16", "wq", abits=16)
            run("rns_w8a8", "rns", code=G.RNSCode(8, 4, 2, abits=8, seed=1))
            run("rns_w8a16", "rns", abits=16, code=G.RNSCode(8, 4, 2, abits=16, seed=2), win=q)
            byz = G.RNSCode(8, 4, 2, abits=8, seed=1); byz.byzantine = {3}
            run("byzantine_rs", "rns", code=byz, win=q)
            naive = G.RNSCode(8, 4, 2, abits=8, seed=1); naive.byzantine = {3}
            naive.verify = False; naive.check_exact = False
            run("byzantine_naive", "rns", code=naive, win=q)
            run("w8a8_on_quarter", "wq", win=q)
            run("w8a16_on_quarter", "wq", abits=16, win=q)
            del run, model, state; gc.collect(); torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            res[name] = {"error": "out of GPU memory", **res.get(name, {})}; log("OOM", name)
            gc.collect(); torch.cuda.empty_cache()
        except Exception:
            res.setdefault(name, {})["error"] = traceback.format_exc()[-1500:]
            log("ERROR quality", name, traceback.format_exc()[-600:])
            gc.collect(); torch.cuda.empty_cache()
    return res


def jdump_partial(obj):
    G.jdump(obj, f"{PREFIX}results3_{ARGS.tag}_quality.partial.json")


STAGES = {"quality": stage_quality, "timing": G.stage_timing, "tasks": G.stage_tasks,
          "attention": G.stage_attention, "generate": G.stage_generate, "privacy": G.stage_privacy}
ARGS = None
PREFIX = ""


def main():
    global ARGS, PREFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--stages", nargs="+", default=["quality"])
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-7B"])
    ap.add_argument("--drop-bf16", nargs="*", default=["Qwen/Qwen2.5-14B"])
    ap.add_argument("--analog", action="store_true")
    ap.add_argument("--windows", type=int, default=20)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--lambada", type=int, default=500)
    ap.add_argument("--task-n", type=int, default=500)
    ap.add_argument("--w8a16-task-n", type=int, default=200)
    ap.add_argument("--coded-task-n", type=int, default=50)
    ap.add_argument("--attn-windows", type=int, default=4)
    ap.add_argument("--attn-seq", type=int, default=512)
    ap.add_argument("--gen-n", type=int, default=100)
    ap.add_argument("--gen-w8a16-n", type=int, default=40)
    ap.add_argument("--gen-coded-n", type=int, default=10)
    ap.add_argument("--gen-max", type=int, default=200)
    ap.add_argument("--priv-model", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    # cpd_gpu2 stages read these names
    args.task_models = args.attn_models = args.gen_models = args.models
    args.priv_model = args.priv_model or args.models[0]
    if args.quick:
        args.models = args.task_models = args.attn_models = args.gen_models = ["Qwen/Qwen2.5-0.5B", "allenai/OLMo-2-0425-1B"]
        args.priv_model = "allenai/OLMo-2-0425-1B"
        args.drop_bf16 = ["Qwen/Qwen2.5-0.5B"]          # exercise the drop path on a small model
        args.windows, args.seq, args.lambada = 2, 256, 10
        args.task_n, args.w8a16_task_n, args.coded_task_n = 4, 4, 2
        args.attn_windows, args.attn_seq = 1, 64
        args.gen_n, args.gen_w8a16_n, args.gen_coded_n, args.gen_max = 1, 1, 1, 16
        args.analog = True
        PREFIX = "quick_"
        if args.stages == ["quality"]:
            args.stages = ["quality", "tasks", "attention", "generate", "privacy"]
    ARGS = args
    G.QUICK = args.quick
    env = {"torch": torch.__version__, "cuda": torch.version.cuda,
           "device": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu", "args": vars(args)}
    try:
        import transformers, datasets
        env["transformers"] = transformers.__version__; env["datasets"] = datasets.__version__
    except Exception:
        pass
    log("environment", env)
    for st in args.stages:
        path = f"{PREFIX}results3_{args.tag}_{st}.json"
        if os.path.exists(path) and not args.force:
            log(f"{path} exists, skipping"); continue
        log(f"===== {args.tag}: stage {st} =====")
        t = time.time()
        try:
            r = STAGES[st](args)
            G.jdump({"result": r, "env": env, "seconds": time.time() - t}, path)
            log(f"{args.tag}: stage {st} finished in {time.time() - t:.0f}s -> {path}")
        except Exception:
            log(f"{args.tag}: stage {st} FAILED:", traceback.format_exc()[-1500:])
            G.jdump({"error": traceback.format_exc()[-3000:], "env": env}, f"{PREFIX}results3_{args.tag}_{st}.FAILED.json")
        gc.collect(); torch.cuda.empty_cache()
    log(f"{args.tag}: ALL DONE")


if __name__ == "__main__":
    main()
