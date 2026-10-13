"""Perplexity (WikiText-2 test, 512-token windows) and LAMBADA last-word accuracy for a given executor.

usage: python evalppl.py CONFIG [--windows N] [--lambada N]
CONFIG examples: fp32 | bf16 | fp16 | w8a8 | w8a12 | w8a16 | ff:N,k,T,abits | analog:fmt,beta
"""
import sys, json, time, argparse, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from gpt2np import GPT2, Tokenizer, log_softmax, plain_linear
from coded import int_reference_linear, plain_fmt_linear, FFCode, AnalogCode

A = os.environ.get("CPD_ASSETS", "/home/claude/assets")
OUT = os.environ.get("CPD_OUT", "/home/claude/code/results")


def make_executor(cfg):
    if cfg == "fp32":
        return plain_linear, None
    if cfg in ("bf16", "fp16"):
        return plain_fmt_linear(cfg), None
    if cfg.startswith("w8a"):
        return int_reference_linear(abits=int(cfg[3:])), None
    if cfg.startswith("ff:"):
        N, k, T, ab = map(int, cfg[3:].split(","))
        c = FFCode(N, k, T, abits=ab); c.check_exact = True
        return c.linear, c
    if cfg.startswith("analog:"):
        fmt, beta = cfg[7:].split(",")
        E = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "realcode_8_4_2.npy"))
        c = AnalogCode(8, 4, 2, fmt=fmt, beta=float(beta), E=E)
        return c.linear, c
    raise ValueError(cfg)


def wikitext_windows(tok, n, L=512):
    cache = os.path.join(A, "wt2_test_ids.npy")
    if not os.path.exists(cache):
        np.save(cache, np.array(tok.encode(open(os.path.join(A, "test.txt")).read()), dtype=np.int64))
    ids = np.load(cache)
    nwin = len(ids) // L
    sel = np.linspace(0, nwin - 1, n).astype(int) if n < nwin else np.arange(nwin)
    return [ids[i * L:(i + 1) * L] for i in sel]


def lambada_items(tok, n):
    items = [json.loads(l)["text"] for l in open(os.path.join(A, "lambada_test.jsonl"))]
    sel = items[:n]
    out = []
    for t in sel:
        ctx, last = t.rsplit(" ", 1)
        c, w = tok.encode(ctx), tok.encode(" " + last)
        out.append((np.array(c + w), len(w)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg"); ap.add_argument("--windows", type=int, default=40); ap.add_argument("--lambada", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tok, g = Tokenizer(), GPT2()
    lin, ctx = make_executor(a.cfg)
    res = {"cfg": a.cfg}
    t0 = time.time()
    if a.windows:
        nll, cnt, bad = 0.0, 0, 0
        for w in wikitext_windows(tok, a.windows):
            with np.errstate(all="ignore"):
                lp = log_softmax(g.forward(w, linear=lin).astype(np.float64))
            v = -lp[np.arange(len(w) - 1), w[1:]]
            if not np.all(np.isfinite(v)):
                bad += 1; v = np.nan_to_num(v, nan=np.log(50257) * 10, posinf=np.log(50257) * 10)
            nll += v.sum(); cnt += len(v)
        res.update(ppl=float(np.exp(nll / cnt)), tokens=cnt, nonfinite_windows=bad)
    if a.lambada:
        hit = 0
        for ids, nw in lambada_items(tok, a.lambada):
            with np.errstate(all="ignore"):
                logits = g.forward(ids[:-1], linear=lin)
            pred = logits[-nw:].argmax(-1)
            hit += int(np.array_equal(pred, ids[-nw:]))
        res.update(lambada_acc=hit / a.lambada, lambada_n=a.lambada)
    if ctx is not None and hasattr(ctx, "log") and ctx.log and "exact" in ctx.log[0]:
        res["ff_ops"] = len(ctx.log); res["ff_all_exact"] = all(r.get("exact") for r in ctx.log)
        res["primes"] = sorted({r["primes"] for r in ctx.log})
    if ctx is not None and hasattr(ctx, "overflow"):
        res["overflow"] = bool(ctx.overflow)
    res["sec"] = time.time() - t0
    print(json.dumps(res))
    with open(os.path.join(OUT, "ppl_" + a.cfg.replace(":", "_").replace(",", "_") + f"_w{a.windows}_l{a.lambada}.json"), "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
