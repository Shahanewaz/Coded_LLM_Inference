"""
GPU experiments for "Quantize the Secret, Not the Share" (RNS-coded LLM prefill).

Runs on one NVIDIA GPU (tested target: RTX 5070, 12 GB) with Hugging Face models.
All N "workers" are emulated on the same GPU, one after another; every worker GEMM is a
real INT8 x INT8 -> INT32 tensor-core GEMM (torch._int_mm), so exactness is checked on real
hardware arithmetic.

What it measures (everything is written to results_gpu.json, progress to the screen):
  1. quality   : WikiText-2 perplexity (+ LAMBADA accuracy) for BF16, uncoded W8A8,
                 RNS-coded W8A8 (N=8, k=4, T=2; bit-exactness checked on every coded GEMM),
                 and the analog baseline (BF16 and FP32 shares, several mask ratios beta).
  2. byzantine : one worker tampers with every coded GEMM; Reed-Solomon check must flag and
                 remove it, and the output must equal the clean run bit for bit.
  3. timing    : real GEMM throughput on this GPU (BF16, FP32, INT8) and the time of one
                 RNS-coded worker vs. a local BF16 GEMM, at Llama-3-8B / Qwen2.5-7B layer shapes.

Usage (from PowerShell):
  & $HOME\\cpd-env\\Scripts\\python.exe cpd_gpu.py            # full run (~30-90 min)
  & $HOME\\cpd-env\\Scripts\\python.exe cpd_gpu.py --quick    # 2-minute smoke test
"""
import argparse, json, math, os, sys, time, itertools, platform, traceback
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")   # harmless Windows cache warning

import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PRIMES = [251, 241, 239, 233, 229, 227, 223, 211]          # all < 256: centered residues fit int8
torch.backends.cuda.matmul.allow_tf32 = False             # FP32 means real FP32
torch.backends.cudnn.allow_tf32 = False

RESULTS = {}
OUTFILE = "results_gpu.json"


def save():
    with open(OUTFILE, "w") as f:
        json.dump(RESULTS, f, indent=1)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ============================================================ exact int8 GEMM (worker kernel)
_INT_MM_OK = None


def int8_gemm(A, B):
    """A: [m, d] int8, B: [d, n] int8 -> exact int32/int64 product [m, n] (as int64).
    Uses tensor-core torch._int_mm when available; otherwise an exact chunked fallback."""
    global _INT_MM_OK
    m = A.shape[0]
    if _INT_MM_OK is not False and A.is_cuda:
        try:
            mp = max(24, m + (-m) % 8)
            Ap = F.pad(A, (0, 0, 0, mp - m)) if mp != m else A
            out = torch._int_mm(Ap.contiguous(), B)[:m]
            _INT_MM_OK = True
            return out.to(torch.int64)
        except Exception as e:  # pragma: no cover
            if _INT_MM_OK is None:
                log("torch._int_mm unavailable, using exact fallback:", repr(e)[:120])
            _INT_MM_OK = False
    # exact fallback: fp32 products of chunks with |partial sum| < 2^24, summed in int64
    d = A.shape[1]
    chunk = max(1, (1 << 24) // (128 * 128))
    out = torch.zeros(m, B.shape[1], dtype=torch.int64, device=A.device)
    Af, Bf = A.float(), B.float()
    for s in range(0, d, chunk):
        out += (Af[:, s:s + chunk] @ Bf[s:s + chunk]).round().to(torch.int64)
    return out


# ============================================================ quantization
def quant_rows(X, bits):
    qmax = 2 ** (bits - 1) - 1
    s = X.abs().amax(dim=1, keepdim=True).float().clamp_min(1e-12) / qmax
    Xq = torch.clamp(torch.round(X.float() / s), -qmax, qmax)
    return Xq, s


def quant_cols(W, bits=8):
    qmax = 2 ** (bits - 1) - 1
    s = W.abs().amax(dim=0, keepdim=True).float().clamp_min(1e-12) / qmax
    Wq = torch.clamp(torch.round(W.float() / s), -qmax, qmax)
    return Wq, s


def primes_for(bound):
    out, prod = [], 1
    for p in PRIMES:
        out.append(p); prod *= p
        if prod > 2 * bound:
            return out
    raise ValueError("bound too large")


def inv_mod(a, p):
    return pow(int(a) % p, p - 2, p)


def mat_inv_mod(A, p):
    n = len(A)
    M = [[int(A[i][j]) % p for j in range(n)] + [int(i == j) for j in range(n)] for i in range(n)]
    for c in range(n):
        piv = next(r for r in range(c, n) if M[r][c] % p)
        M[c], M[piv] = M[piv], M[c]
        iv = inv_mod(M[c][c], p)
        M[c] = [(x * iv) % p for x in M[c]]
        for r in range(n):
            if r != c and M[r][c]:
                f = M[r][c]
                M[r] = [(x - f * y) % p for x, y in zip(M[r], M[c])]
    return [row[n:] for row in M]



def mix(V, U, p):
    """out[i] = sum_j V[i, j] * U[j]  (mod p) for a small int64 matrix V and stacked int64 tensor U.
    Written with elementwise ops because CUDA has no int64 matrix multiply."""
    V = V.to(U.device)
    out = torch.zeros((V.shape[0],) + tuple(U.shape[1:]), dtype=torch.int64, device=U.device)
    shape = (-1,) + (1,) * (U.dim() - 1)
    for j in range(V.shape[1]):
        out += V[:, j].reshape(shape) * U[j]
    return out % p


# ============================================================ RNS secret-shared polynomial code
class RNSCode:
    def __init__(self, N=8, k=4, T=2, abits=8, seed=0):
        self.N, self.k, self.T, self.abits = N, k, T, abits
        self.gen = torch.Generator(device=DEV); self.gen.manual_seed(seed)
        self.cache = {}
        self.byzantine = None          # None or worker index that tampers
        self.verify = True
        self.stats = {"gemms": 0, "exact": 0, "flagged": 0, "corrected": 0}
        self.check_exact = True

    def tables(self, p):
        if p not in self.cache:
            KT = self.k + self.T
            V = [[pow(a, j, p) for j in range(KT)] for a in range(1, self.N + 1)]
            self.cache[p] = {"V": torch.tensor(V, dtype=torch.int64, device=DEV), "Vl": V}
        return self.cache[p]

    def decode_matrix(self, p, rows):
        key = (p, tuple(rows))
        if key not in self.cache:
            V = self.tables(p)["Vl"]
            self.cache[key] = torch.tensor(mat_inv_mod([V[i] for i in rows], p), dtype=torch.int64, device=DEV)
        return self.cache[key]

    def consistent(self, Zs, primes, rows):
        KT = self.k + self.T
        base, rest = rows[:KT], rows[KT:]
        for p, Z in zip(primes, Zs):
            coef = mix(self.decode_matrix(p, base), Z[base] % p, p)
            pred = mix(self.tables(p)["V"][rest], coef, p)
            if not torch.equal(pred, Z[rest] % p):
                return False
        return True

    def linear(self, X, Wq, sw, bias):
        """X [m, d] (any float dtype), Wq [d, n] int8 (public), sw [1, n]."""
        N, k, T = self.N, self.k, self.T
        Xq, sx = quant_rows(X, self.abits)
        m, d = Xq.shape
        bound = (2 ** (self.abits - 1) - 1) * 127 * d
        primes = primes_for(bound)
        b = -(-m // k)
        Xp = torch.zeros(k * b, d, dtype=torch.int64, device=DEV)
        Xp[:m] = Xq.to(torch.int64)
        blocks = Xp.view(k, b, d)
        Zs = []
        for p in primes:
            masks = torch.randint(0, p, (T, b, d), generator=self.gen, device=DEV, dtype=torch.int64)
            U = torch.cat([blocks % p, masks], 0)                                  # [k+T, b, d]
            S = mix(self.tables(p)["V"], U, p)                # [N, b, d]
            S = torch.where(S > p // 2, S - p, S).to(torch.int8)                   # centered int8 shares
            # ---- workers: one INT8 GEMM each (batched across workers for speed: same math)
            Z = int8_gemm(S.reshape(N * b, d), Wq).view(N, b, -1) % p             # worker reduces mod p
            if self.byzantine is not None:
                Z[self.byzantine] = (Z[self.byzantine] + 1 + (Z[self.byzantine] % 7)) % p
            Zs.append(Z)
        self.stats["gemms"] += 1
        avail = list(range(N))
        bad = []
        if self.verify and not self.consistent(Zs, primes, avail):
            self.stats["flagged"] += 1
            red = N - (k + T)
            for a in range(1, red // 2 + 1):
                for cand in itertools.combinations(avail, a):
                    rest = [i for i in avail if i not in cand]
                    if self.consistent(Zs, primes, rest):
                        bad = list(cand); break
                if bad:
                    break
            if not bad:
                raise RuntimeError("uncorrectable tampering detected: abort")
            self.stats["corrected"] += 1
        use = [i for i in avail if i not in bad][: k + T]
        # ---- decode each prime, then CRT
        M = 1
        for p in primes:
            M *= p
        acc = None
        for p, Z in zip(primes, Zs):
            coef = mix(self.decode_matrix(p, use)[:k], Z[use], p)     # [k, b, n]
            Mi = M // p
            c = (Mi * inv_mod(Mi % p, p)) % M
            term = (coef * c) % M
            acc = term if acc is None else (acc + term) % M
        Yint = torch.where(acc > M // 2, acc - M, acc).reshape(k * b, -1)[:m]
        if self.check_exact:
            ref = reference_int(Xq, Wq, self.abits)
            self.stats["exact"] += int(torch.equal(Yint, ref))
        Y = Yint.double() * sx.double() * sw.double()
        if bias is not None:
            Y = Y + bias.double()
        return Y.to(X.dtype)


def reference_int(Xq, Wq, abits):
    if abits <= 8:
        return int8_gemm(Xq.to(torch.int8), Wq)
    # wider activations: exact limb decomposition Xq = sum_j 128^j * l_j with |l_j| <= 64
    Xi = Xq.to(torch.int64)
    out, scale = None, 1
    while bool((Xi != 0).any()):
        lo = ((Xi + 64) % 128) - 64
        part = int8_gemm(lo.to(torch.int8), Wq) * scale
        out = part if out is None else out + part
        Xi = (Xi - lo) // 128
        scale *= 128
    return out if out is not None else torch.zeros(Xq.shape[0], Wq.shape[1], dtype=torch.int64, device=Xq.device)


# ============================================================ analog baseline
def make_real_code(N=8, k=4, T=2, seed=0, iters=600, restarts=3):
    g = torch.Generator().manual_seed(seed)
    def metrics(E):
        amp = max(torch.linalg.matrix_norm(torch.linalg.pinv(E[list(S)])[:k], 2).item()
                  for S in itertools.combinations(range(N), k + T))
        gain = max(torch.linalg.matrix_norm(torch.linalg.solve(E[list(S), k:], E[list(S), :k]), 2).item() ** 2
                   for S in itertools.combinations(range(N), T))
        return amp, gain
    best, bv = None, float("inf")
    for _ in range(restarts):
        E = torch.randn(N, k + T, generator=g, dtype=torch.float64); E /= E.norm(dim=1, keepdim=True)
        a, gg = metrics(E); v = a * math.sqrt(gg); step = 0.3
        for it in range(iters):
            Fm = E + step * torch.randn(E.shape, generator=g, dtype=torch.float64); Fm /= Fm.norm(dim=1, keepdim=True)
            a2, g2 = metrics(Fm); v2 = a2 * math.sqrt(g2)
            if v2 < v:
                E, v = Fm, v2
            elif it % 100 == 99:
                step *= 0.7
        if v < bv:
            best, bv = E, v
    return best


def worst_leak_per_entry(E, k, T, beta):
    N = E.shape[0]; worst = 0.0
    for S in itertools.combinations(range(N), T):
        A = torch.linalg.solve(E[list(S), k:], E[list(S), :k])
        val = 0.5 * torch.logdet(torch.eye(T, dtype=torch.float64) + A @ A.T / beta ** 2).item()
        worst = max(worst, val)
    return worst


class AnalogCode:
    def __init__(self, E, k, T, fmt, beta, seed=0):
        self.E = E.to(DEV); self.k, self.T, self.fmt, self.beta = k, T, fmt, beta
        self.D = torch.linalg.pinv(self.E)[:k]
        self.gen = torch.Generator(device=DEV); self.gen.manual_seed(seed)
        self.overflow = False

    def linear(self, X, W, bias):
        k, T, E = self.k, self.T, self.E
        N = E.shape[0]
        Xd = X.double()
        m, d = Xd.shape
        b = -(-m // k)
        Xp = torch.zeros(k * b, d, dtype=torch.float64, device=DEV); Xp[:m] = Xd
        rms = Xd.pow(2).mean(0).sqrt() + 1e-12
        R = torch.randn((T, b, d), generator=self.gen, device=DEV, dtype=torch.float64) * (self.beta * rms)
        U = torch.cat([Xp.view(k, b, d), R], 0)
        S = torch.tensordot(E, U, dims=1)                       # [N, b, d] float64
        dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[self.fmt]
        Sw = S.to(dt).reshape(N * b, d)
        Z = (Sw @ W.to(dt)).view(N, b, -1).double()            # tensor-core GEMM in the worker format
        if not torch.isfinite(Z).all():
            self.overflow = True
        Y = torch.tensordot(self.D, Z, dims=1).reshape(k * b, -1)[:m]
        if bias is not None:
            Y = Y + bias.double()
        return Y.to(X.dtype)


# ============================================================ model surgery
class CodedLinear(nn.Module):
    def __init__(self, lin: nn.Linear, state):
        super().__init__()
        self.lin, self.state = lin, state
        W = lin.weight.data.t()                              # [d, n]
        Wq, sw = quant_cols(W)
        # column-major int8 weight (the layout cuBLASLt INT8 GEMMs expect)
        self.register_buffer("Wq", Wq.to(torch.int8).t().contiguous().t(), persistent=False)
        self.register_buffer("sw", sw, persistent=False)

    def forward(self, x):
        mode = self.state["mode"]
        if mode == "plain":
            return self.lin(x)
        shp = x.shape
        X = x.reshape(-1, shp[-1])
        b = self.lin.bias
        if mode == "w8a8":
            Xq, sx = quant_rows(X, self.state.get("abits", 8))
            Yint = reference_int(Xq, self.Wq, self.state.get("abits", 8))
            Y = (Yint.double() * sx.double() * self.sw.double())
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


def wrap_model(model, state):
    n = 0
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(parent, name, CodedLinear(child, state)); n += 1
    return n


# ============================================================ data / evaluation
def load_wikitext(tok, n_windows, L):
    from datasets import load_dataset
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    nwin = len(ids) // L
    sel = torch.linspace(0, nwin - 1, min(n_windows, nwin)).long().tolist()
    return [ids[i * L:(i + 1) * L] for i in sel], int(len(ids))


def load_lambada(tok, n):
    from datasets import load_dataset
    try:
        ds = load_dataset("EleutherAI/lambada_openai", "default", split="test")
    except Exception:
        try:
            ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
        except Exception as e:
            log("LAMBADA unavailable, skipping:", repr(e)[:150])
            return []
    out = []
    for t in ds["text"][:n]:
        ctx, last = t.rsplit(" ", 1)
        c = tok(ctx).input_ids; w = tok(" " + last).input_ids
        out.append((torch.tensor(c + w), len(w)))
    return out


@torch.no_grad()
def ppl(model, windows):
    nll, cnt, bad = 0.0, 0, 0
    for w in windows:
        ids = w.unsqueeze(0).to(DEV)
        logits = model(ids).logits[0, :-1].float()
        v = F.cross_entropy(logits, ids[0, 1:], reduction="none")
        if not torch.isfinite(v).all():
            bad += 1; v = torch.nan_to_num(v, nan=30.0, posinf=30.0)
        nll += v.sum().item(); cnt += v.numel()
    return math.exp(nll / cnt), cnt, bad


@torch.no_grad()
def lambada_acc(model, items):
    hit = 0
    for ids, nw in items:
        x = ids[:-1].unsqueeze(0).to(DEV)
        pred = model(x).logits[0, -nw:].argmax(-1).cpu()
        hit += int(torch.equal(pred, ids[-nw:]))
    return hit / max(1, len(items))


# ============================================================ timing
def bench(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


@torch.no_grad()
def timing(quick):
    if DEV != "cuda":
        return {}
    shapes = {"Llama-3-8B gate/up (m x 4096 x 14336)": (4096, 14336),
              "Llama-3-8B down (m x 14336 x 4096)": (14336, 4096),
              "Qwen2.5-7B up (m x 3584 x 18944)": (3584, 18944)}
    ms = [2048] if quick else [2048, 8192]
    out = {}
    N, k, T = 8, 4, 2
    for m in ms:
        for name, (d, n) in shapes.items():
            key = f"{name}, m={m}"
            try:
                A16 = torch.randn(m, d, device=DEV, dtype=torch.bfloat16); B16 = torch.randn(d, n, device=DEV, dtype=torch.bfloat16)
                A32, B32 = A16.float(), B16.float()
                A8 = torch.randint(-125, 126, (m, d), device=DEV, dtype=torch.int8)
                B8 = torch.randint(-127, 128, (n, d), device=DEV, dtype=torch.int8).t()   # column-major
                W8 = torch.randint(-125, 126, (4 * (m // k), d), device=DEV, dtype=torch.int8)   # r=4 residue shares, one worker
                flops = 2 * m * d * n
                t_bf16 = bench(lambda: A16 @ B16)
                t_fp32 = bench(lambda: A32 @ B32, reps=5)
                t_int8 = bench(lambda: torch._int_mm(A8, B8))
                t_worker = bench(lambda: torch._int_mm(W8, B8) % 251)
                # coordinator encode for one prime (all N shares) + decode, on this GPU
                X = torch.randint(-127, 128, (m, d), device=DEV, dtype=torch.int64)
                V = torch.randint(0, 251, (N, k + T), device=DEV, dtype=torch.int64)
                def enc():
                    U = torch.cat([(X % 251).view(k, m // k, d), torch.randint(0, 251, (T, m // k, d), device=DEV)], 0)
                    return mix(V, U, 251).to(torch.int8)
                t_enc = 4 * bench(enc, reps=5)
                Zw = torch.randint(0, 251, (k + T, m // k, n), device=DEV, dtype=torch.int64)
                Dm = torch.randint(0, 251, (k, k + T), device=DEV, dtype=torch.int64)
                t_dec = 4 * bench(lambda: mix(Dm, Zw, 251), reps=5)
                out[key] = {"bf16_ms": t_bf16 * 1e3, "fp32_ms": t_fp32 * 1e3, "int8_ms": t_int8 * 1e3,
                            "bf16_TFLOPS": flops / t_bf16 / 1e12, "fp32_TFLOPS": flops / t_fp32 / 1e12,
                            "int8_TOPS": flops / t_int8 / 1e12,
                            "rns_worker_ms(r=4,k=4)": t_worker * 1e3, "rns_worker_vs_local_bf16": t_worker / t_bf16,
                            "analog_fp32_worker_vs_local_bf16": (t_fp32 / k) / t_bf16,
                            "coordinator_encode_ms(4 primes)": t_enc * 1e3, "coordinator_decode_ms(4 primes)": t_dec * 1e3}
                log("timing", key, {a: round(b, 3) for a, b in out[key].items()})
                del A16, B16, A32, B32, A8, B8, W8, X, Zw
                torch.cuda.empty_cache()
            except Exception as e:
                out[key] = {"error": repr(e)[:300]}
                log("timing failed", key, repr(e)[:200])
                torch.cuda.empty_cache()
    return out


# ============================================================ main
def run_model(name, args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log("=== model", name)
    tok = AutoTokenizer.from_pretrained(name)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16)
    model = model.to(DEV).eval()
    res = RESULTS.setdefault("models", {}).setdefault(name, {})
    windows, ntok = load_wikitext(tok, args.windows, args.seq)
    res["wikitext"] = {"windows": len(windows), "seq": args.seq, "test_tokens_total": ntok}
    lam = load_lambada(tok, args.lambada) if args.lambada else []
    state = {"mode": "plain"}
    nlin = wrap_model(model, state)
    res["coded_linear_layers"] = nlin
    log(f"wrapped {nlin} linear layers; {len(windows)} windows of {args.seq} tokens")

    def run(tag, setup, use_lambada=False, win=None):
        win = windows if win is None else win
        setup()
        t = time.time()
        p, cnt, bad = ppl(model, win)
        entry = {"ppl": p, "tokens": cnt, "nonfinite_windows": bad, "sec": time.time() - t}
        if use_lambada and lam:
            entry["lambada_acc"] = lambada_acc(model, lam); entry["lambada_n"] = len(lam)
        if state.get("code") is not None and isinstance(state["code"], RNSCode):
            entry.update({"coded_gemms": state["code"].stats["gemms"], "exact_gemms": state["code"].stats["exact"],
                          "flagged": state["code"].stats["flagged"], "corrected": state["code"].stats["corrected"]})
        if state.get("code") is not None and isinstance(state["code"], AnalogCode):
            entry["overflow"] = state["code"].overflow
        res[tag] = entry
        log(tag, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in entry.items()})
        save()

    def set_mode(mode, code=None, abits=8):
        def f():
            state.clear(); state.update({"mode": mode, "code": code, "abits": abits})
        return f

    run("bf16", set_mode("plain"), use_lambada=True)
    run("w8a8", set_mode("w8a8"), use_lambada=True)
    run("rns_w8a8", set_mode("rns", RNSCode(8, 4, 2, abits=8, seed=1)))
    run("w8a16", set_mode("w8a8", abits=16))
    run("rns_w8a16", set_mode("rns", RNSCode(8, 4, 2, abits=16, seed=2)), win=windows[:max(1, len(windows) // 4)])
    # Byzantine: worker 3 tampers with every coded GEMM
    byz = RNSCode(8, 4, 2, abits=8, seed=1); byz.byzantine = 3
    run("rns_w8a8_byzantine_rs", set_mode("rns", byz), win=windows[:max(1, len(windows) // 4)])
    naive = RNSCode(8, 4, 2, abits=8, seed=1); naive.byzantine = 3; naive.verify = False; naive.check_exact = False
    run("rns_w8a8_byzantine_naive", set_mode("rns", naive), win=windows[:max(1, len(windows) // 4)])
    # analog baseline
    E = make_real_code() if args.quick is False else make_real_code(iters=60, restarts=1)
    asub = windows[:max(1, len(windows) // 2)]
    n_entries = None
    for fmt, beta in ([("bf16", 10.0), ("fp32", 1e4)] if args.quick else
                      [("bf16", 1.0), ("bf16", 10.0), ("bf16", 100.0), ("fp16", 100.0), ("fp16", 1000.0), ("fp32", 1e3), ("fp32", 1e4)]):
        code = AnalogCode(E, 4, 2, fmt, beta, seed=3)
        run(f"analog_{fmt}_beta{beta:g}", set_mode("analog", code), win=asub)
        res[f"analog_{fmt}_beta{beta:g}"]["leak_per_entry_worst_pair_nats"] = worst_leak_per_entry(E, 4, 2, beta)
    run("bf16_on_analog_subset", set_mode("plain"), win=asub)
    run("w8a8_on_analog_subset", set_mode("w8a8"), win=asub)
    save()
    del model
    torch.cuda.empty_cache()


def main():
    global OUTFILE
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B"])
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--lambada", type=int, default=1000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument("--out", default="results_gpu.json")
    args = ap.parse_args()
    OUTFILE = args.out
    if args.quick:
        args.models = args.models[:1] if args.models != ["Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B"] else ["Qwen/Qwen2.5-0.5B"]
        args.windows, args.seq, args.lambada = 2, 256, 20
    RESULTS["env"] = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
                      "python": sys.version.split()[0], "platform": platform.platform(), "args": vars(args)}
    log("environment", RESULTS["env"])
    save()
    if not args.skip_timing:
        RESULTS["timing"] = timing(args.quick); save()
    for name in args.models:
        try:
            run_model(name, args)
        except torch.cuda.OutOfMemoryError as e:
            RESULTS.setdefault("errors", {})[name] = "out of GPU memory"; save()
            log("OUT OF MEMORY on", name, "- skipping it")
            torch.cuda.empty_cache()
        except Exception as e:
            RESULTS.setdefault("errors", {})[name] = traceback.format_exc()[-2000:]; save()
            log("ERROR on", name, repr(e)[:300])
    log("DONE. Results written to", os.path.abspath(OUTFILE))


if __name__ == "__main__":
    main()
