"""
Overnight GPU experiments for "Quantize the Secret, Not the Share" (part 2).

Stages (each writes results2_<stage>.json and is skipped on re-run if that file already exists):
  timing     fast RNS coordinator (exact FP32-GEMM encode/decode) + worker timings
  stress     exactness / straggler / Byzantine sweeps over (N, k, T), random layer shapes
  tasks      zero-shot multiple-choice benchmarks (ARC-E/C, HellaSwag, BoolQ, WinoGrande, PIQA, MMLU)
             for BF16, W8A8, W8A16 and RNS-coded (subset, must match W8A8 exactly)
  attention  coded attention (Theorem 4): QK^T and PV both coded, exactness + perplexity
  generate   GSM8K greedy generation; RNS-coded prefill AND decode must reproduce W8A8 tokens exactly
  privacy    neural inversion attack on Qwen hidden states incl. canary secrets (PINs)

Usage (PowerShell, inside Documents\\cpd_gpu):
  & $HOME\\cpd-env\\Scripts\\python.exe cpd_gpu2.py --quick        # ~5 min self-test
  & $HOME\\cpd-env\\Scripts\\python.exe cpd_gpu2.py                # everything (~6-8 h)
  & $HOME\\cpd-env\\Scripts\\python.exe cpd_gpu2.py --stages tasks # just one stage
Progress is printed and also appended to cpd_run2.log.
"""
import argparse, json, math, os, sys, time, itertools, platform, traceback, random, re
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PRIMES = [251, 241, 239, 233, 229, 227, 223, 211]
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
LOGFILE = "cpd_run2.log"
QUICK = False


def log(*a):
    msg = time.strftime("%H:%M:%S") + " " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    try:
        with open(LOGFILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def jdump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


# ============================================================ exact int8 GEMM
_INT_MM_OK = None


def int8_gemm(A, B):
    """Exact A @ B for int8 A [m, d], int8 B [d, n]; returns int64. Tensor-core _int_mm when possible."""
    global _INT_MM_OK
    m = A.shape[0]
    if _INT_MM_OK is not False and A.is_cuda:
        try:
            mp = max(24, m + (-m) % 8)
            Ap = F.pad(A, (0, 0, 0, mp - m)) if mp != m else A
            if B.shape[1] % 8 or A.shape[1] % 8:
                raise RuntimeError("shape not multiple of 8")
            out = torch._int_mm(Ap.contiguous(), B)[:m]
            _INT_MM_OK = True
            return out.to(torch.int64)
        except Exception as e:
            if _INT_MM_OK is None:
                log("note: torch._int_mm not used for a shape, exact fallback:", repr(e)[:100])
    d = A.shape[1]
    chunk = max(1, (1 << 24) // (128 * 128))
    out = torch.zeros(m, B.shape[1], dtype=torch.int64, device=A.device)
    Af, Bf = A.float(), B.float()
    for s in range(0, d, chunk):
        out += (Af[:, s:s + chunk] @ Bf[s:s + chunk]).round().to(torch.int64)
    return out


# ============================================================ quantization helpers
def quant_rows(X, bits):
    qmax = 2 ** (bits - 1) - 1
    s = X.abs().amax(dim=-1, keepdim=True).float().clamp_min(1e-12) / qmax
    return torch.clamp(torch.round(X.float() / s), -qmax, qmax), s


def quant_cols(W, bits=8):
    qmax = 2 ** (bits - 1) - 1
    s = W.abs().amax(dim=0, keepdim=True).float().clamp_min(1e-12) / qmax
    return torch.clamp(torch.round(W.float() / s), -qmax, qmax), s


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


def int_ref(Xq, Wq, abits):
    """Exact Xq @ Wq for integer Xq of any width (limb split into int8 pieces)."""
    if abits <= 8:
        return int8_gemm(Xq.to(torch.int8), Wq)
    Xi = Xq.to(torch.int64)
    out, scale = None, 1
    while bool((Xi != 0).any()):
        lo = ((Xi + 64) % 128) - 64
        part = int8_gemm(lo.to(torch.int8), Wq) * scale
        out = part if out is None else out + part
        Xi = (Xi - lo) // 128
        scale *= 128
    return out if out is not None else torch.zeros(Xq.shape[0], Wq.shape[1], dtype=torch.int64, device=Xq.device)


def fmm(V, U2, p):
    """Exact small-matrix x tall-matrix product mod p with an FP32 GEMM.
    V: [a, b] ints in [0,p), U2: [b, L] ints in [0,p) (float32). Requires b*(p-1)^2 < 2^24."""
    assert V.shape[1] * (p - 1) ** 2 < (1 << 24)
    return torch.remainder(V @ U2, p)


def crt(res, primes):
    M = 1
    for p in primes:
        M *= p
    acc = None
    for p, r in zip(primes, res):
        Mi = M // p
        c = (Mi * inv_mod(Mi % p, p)) % M
        term = (r.to(torch.int64) * c) % M
        acc = term if acc is None else (acc + term) % M
    return torch.where(acc > M // 2, acc - M, acc)


# ============================================================ coding tables
class Tables:
    def __init__(self):
        self.c = {}

    def vander(self, p, n_eval, deg1):
        key = ("V", p, n_eval, deg1)
        if key not in self.c:
            self.c[key] = [[pow(a, j, p) for j in range(deg1)] for a in range(1, n_eval + 1)]
        return self.c[key]

    def t(self, key, fn):
        if key not in self.c:
            self.c[key] = torch.tensor(fn(), dtype=torch.float32, device=DEV)
        return self.c[key]


TAB = Tables()


class RNSCode:
    """Polynomial secret sharing over an RNS of primes < 256 (Theorems 2-3), fast exact implementation:
    all coordinator-side mixing is an exact FP32 GEMM, every worker GEMM is INT8 (torch._int_mm)."""

    def __init__(self, N=8, k=4, T=2, abits=8, seed=0, k_decode=1):
        self.N, self.k, self.T, self.abits, self.k_decode = N, k, T, abits, k_decode
        self.gen = torch.Generator(device=DEV); self.gen.manual_seed(seed)
        self.byzantine = set()       # worker indices that tamper
        self.stragglers = set()
        self.verify = True
        self.check_exact = True
        self.stats = {"gemms": 0, "exact": 0, "flagged": 0, "corrected": 0, "aborted": 0}

    # ---- generic engine: shares of a stacked secret [k, b, d] -> worker products -> coefficients
    def _share(self, p, blocks_f, k, T, N):
        """blocks_f: [k, b, d] float32 residues in [0,p). Returns centered int8 shares [N, b, d]."""
        _, b, d = blocks_f.shape
        masks = torch.randint(0, p, (T, b, d), generator=self.gen, device=DEV).float()
        U = torch.cat([blocks_f, masks], 0).reshape(k + T, -1)
        V = TAB.t(("Vs", p, N, k + T), lambda: TAB.vander(p, N, k + T))
        S = fmm(V, U, p).reshape(N, b, d)
        return torch.where(S > p // 2, S - p, S).to(torch.int8)

    def _decode(self, p, Z, n_coef, deg1, N, avail):
        """Z: [N, L] float residues in [0,p). Exact consistency check / error location / interpolation.
        Returns ([n_coef, L] coefficients, list of excluded workers)."""
        def coeffs(rows):
            Dm = TAB.t(("D", p, N, deg1, tuple(rows)), lambda: mat_inv_mod([TAB.vander(p, N, deg1)[i] for i in rows], p))
            return fmm(Dm, Z[rows], p)
        def consistent(rows):
            base, rest = rows[:deg1], rows[deg1:]
            if not rest:
                return True
            c = coeffs(base)
            Vr = TAB.t(("Vr", p, N, deg1, tuple(rest)), lambda: [TAB.vander(p, N, deg1)[i] for i in rest])
            return torch.equal(fmm(Vr, c, p), Z[rest])
        bad = []
        if self.verify and not consistent(avail):
            red = len(avail) - deg1
            for a in range(1, red // 2 + 1):
                for cand in itertools.combinations(avail, a):
                    rest = [i for i in avail if i not in cand]
                    if consistent(rest):
                        bad = list(cand); break
                if bad:
                    break
            if not bad:
                bad = [-1]
        use = [i for i in avail if i not in bad][:deg1]
        if len(use) < deg1 or bad == [-1]:
            return None, bad
        return coeffs(use)[:n_coef], bad

    def _tamper(self, Z, p):
        for w in self.byzantine:
            Z[w] = torch.remainder(Z[w] + 1 + torch.remainder(Z[w], 7), p)
        return Z

    # ---- one-sided: Y = X W, W public (Theorems 2-3)
    def linear(self, X, Wq, sw, bias):
        N, T = self.N, self.T
        Xq, sx = quant_rows(X, self.abits)
        m, d = Xq.shape
        k = self.k if m >= 4 * self.k else self.k_decode           # decode steps (m small): k=1
        primes = primes_for((2 ** (self.abits - 1) - 1) * 127 * d)
        b = -(-m // k)
        Xp = torch.zeros(k * b, d, device=DEV)
        Xp[:m] = Xq
        shares = []
        for p in primes:
            shares.append(self._share(p, torch.remainder(Xp, p).view(k, b, d), k, T, N))
        S = torch.stack(shares, 0)                                     # [r, N, b, d] int8
        Zall = int8_gemm(S.reshape(-1, d), Wq).view(len(primes), N, b, -1)
        avail = [i for i in range(N) if i not in self.stragglers]
        res, flagged, bad_all = [], False, []
        for pi, p in enumerate(primes):
            Z = torch.remainder(Zall[pi], p).float().reshape(N, -1)
            Z = self._tamper(Z, p)
            c, bad = self._decode(p, Z, k, k + self.T, N, avail)
            if bad:
                flagged = True; bad_all = bad
            if c is None:
                self.stats["aborted"] += 1
                raise RuntimeError("tampering detected but not correctable: abort")
            res.append(c.reshape(k, b, -1))
        self.stats["gemms"] += 1
        if flagged:
            self.stats["flagged"] += 1; self.stats["corrected"] += 1
        Yint = crt(res, primes).reshape(k * b, -1)[:m]
        if self.check_exact:
            self.stats["exact"] += int(torch.equal(Yint, int_ref(Xq, Wq, self.abits)))
        Y = Yint.double() * sx.double() * sw.double()
        if bias is not None:
            Y = Y + bias.double()
        return Y.to(X.dtype)

    # ---- two-sided: C = A B^T with BOTH private integer matrices (Theorem 4)
    def private_product(self, Aq, Bq, bound):
        """Aq [m, d] and Bq [n, d] integer tensors (|.| small). Returns exact Aq @ Bq^T (int64).
        A is split into k row blocks: pA(z) = sum_j A_j z^j + sum_t R_t z^(k+t);
        B is shared whole:            pB(z) = B + sum_t R'_t z^(k+t).
        The product has degree 2(k+T)-2 and its first k coefficients are A_j B^T."""
        N, T = self.N, self.T
        m, d = Aq.shape
        n = Bq.shape[0]
        k = self.k if m >= 4 * self.k else 1
        deg1 = 2 * (k + T) - 1
        assert N >= deg1, f"need N >= 2(k+T)-1 = {deg1}"
        primes = primes_for(bound)
        b = -(-m // k)
        Ap = torch.zeros(k * b, d, device=DEV); Ap[:m] = Aq
        res = []
        avail = [i for i in range(N) if i not in self.stragglers]
        for p in primes:
            SA = self._share(p, torch.remainder(Ap, p).view(k, b, d), k, T, N)                      # [N, b, d]
            # B share: coefficients at z^0 (secret) and z^k..z^(k+T-1) (masks)
            maskB = torch.randint(0, p, (T, n, d), generator=self.gen, device=DEV).float()
            UB = torch.cat([torch.remainder(Bq.float(), p).unsqueeze(0), maskB], 0).reshape(1 + T, -1)
            VB = TAB.t(("VB", p, N, k, T), lambda: [[1] + [pow(a, k + t, p) for t in range(T)] for a in range(1, N + 1)])
            SB = fmm(VB, UB, p).reshape(N, n, d)
            SB = torch.where(SB > p // 2, SB - p, SB).to(torch.int8)
            Z = torch.stack([int8_gemm(SA[i], SB[i].t()) for i in range(N)], 0)   # worker i: int8 GEMM
            Z = torch.remainder(Z, p).float().reshape(N, -1)
            Z = self._tamper(Z, p)
            c, bad = self._decode(p, Z, k, deg1, N, avail)
            if c is None:
                raise RuntimeError("attention tampering not correctable: abort")
            res.append(c.reshape(k, b, n))
        return crt(res, primes).reshape(k * b, n)[:m]


# ============================================================ model surgery
class CodedLinear(nn.Module):
    def __init__(self, lin, state):
        super().__init__()
        self.lin, self.state = lin, state
        Wq, sw = quant_cols(lin.weight.data.t())
        self.register_buffer("Wq", Wq.to(torch.int8).t().contiguous().t(), persistent=False)
        self.register_buffer("sw", sw, persistent=False)

    def forward(self, x):
        mode = self.state["mode"]
        if mode == "plain":
            return self.lin(x)
        shp = x.shape
        X = x.reshape(-1, shp[-1])
        b = self.lin.bias
        if mode == "wq":
            ab = self.state["abits"]
            Xq, sx = quant_rows(X, ab)
            Y = int_ref(Xq, self.Wq, ab).double() * sx.double() * self.sw.double()
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


def repeat_kv(x, n):
    if n == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n, s, d).reshape(b, h * n, s, d)


CUR = {"state": {"attn": "plain"}}


def make_attention(orig):
    """Replacement for eager_attention_forward. Modes: 'plain' (original), 'qattn' (int8-quantized
    attention, uncoded reference), 'cattn' (the same arithmetic computed with Theorem-4 coding)."""
    def fn(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        state = CUR["state"]
        am = state.get("attn", "plain")
        if am == "plain":
            return orig(module, query, key, value, attention_mask, scaling, dropout=dropout, **kw)
        ng = getattr(module, "num_key_value_groups", 1)
        K = repeat_kv(key, ng); Vv = repeat_kv(value, ng)
        B, H, m, dh = query.shape
        n = K.shape[2]
        out = torch.empty(B, H, m, dh, device=query.device, dtype=torch.float32)
        code = state.get("acode")
        for bi in range(B):
            for h in range(H):
                Qq, sq = quant_rows(query[bi, h], 8)               # per-token
                Kq, sk = quant_rows(K[bi, h], 8)                   # per-token
                if am == "cattn":
                    S_int = code.private_product(Qq, Kq, 127 * 127 * dh)
                    if state.get("check"):
                        state["att_checked"] += 1
                        state["att_exact"] += int(torch.equal(S_int, int8_gemm(Qq.to(torch.int8), Kq.to(torch.int8).contiguous().t())))
                else:
                    S_int = int8_gemm(Qq.to(torch.int8), Kq.to(torch.int8).contiguous().t())
                scores = S_int.double() * sq.double() * sk.double().t() * scaling
                if attention_mask is not None:
                    scores = scores + attention_mask[bi, 0, :, :n].double()
                P = torch.softmax(scores, dim=-1)
                Pq, sp = quant_rows(P, 8)                          # per-row (non-negative, so [0,127])
                Vt = Vv[bi, h]                                     # [n, dh]; quantize per channel
                Vq, sv = quant_cols(Vt, 8)
                if am == "cattn":
                    O_int = code.private_product(Pq, Vq.t().contiguous(), 127 * 127 * n)
                    if state.get("check"):
                        state["att_checked"] += 1
                        state["att_exact"] += int(torch.equal(O_int, int8_gemm(Pq.to(torch.int8), Vq.to(torch.int8).contiguous().t().contiguous().t())))
                else:
                    O_int = int8_gemm(Pq.to(torch.int8), Vq.to(torch.int8).t().contiguous().t())
                out[bi, h] = (O_int.double() * sp.double() * sv.double()).float()
        return out.to(query.dtype).transpose(1, 2).contiguous(), None
    return fn


def wrap_model(model, state):
    n = 0
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(parent, name, CodedLinear(child, state)); n += 1
    CUR["state"] = state
    mod = sys.modules[type(model.model.layers[0].self_attn).__module__]
    if hasattr(mod, "eager_attention_forward") and not getattr(mod, "_cpd_patched", False):
        mod.eager_attention_forward = make_attention(mod.eager_attention_forward)
        mod._cpd_patched = True
    return n


def load_model(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    kw = dict(attn_implementation="eager")
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, **kw)
    model = model.to(DEV).eval()
    state = {"mode": "plain", "attn": "plain"}
    n = wrap_model(model, state)
    log(f"loaded {name}: {n} linear layers wrapped")
    return tok, model, state


def set_mode(state, mode, code=None, abits=8):
    state.update({"mode": mode, "code": code, "abits": abits})


# ============================================================ datasets
def ds_load(*args, **kw):
    from datasets import load_dataset
    return load_dataset(*args, **kw)


def wikitext_windows(tok, n, L, split="test"):
    try:
        ds = ds_load("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    except Exception:
        ds = ds_load("wikitext", "wikitext-2-raw-v1", split=split)
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    nwin = len(ids) // L
    sel = torch.linspace(0, nwin - 1, min(n, nwin)).long().tolist()
    return [ids[i * L:(i + 1) * L] for i in sel]


def get_tasks(n_per_task):
    """Returns {task: [ (context, [continuations], gold_index, [norm_lengths]) ]} using public HF datasets."""
    tasks = {}
    rng = random.Random(0)

    def take(ds, n):
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        return [ds[i] for i in idx[:n]]

    def safe(name, fn):
        try:
            tasks[name] = fn()
            log(f"task {name}: {len(tasks[name])} examples")
        except Exception as e:
            log(f"task {name} unavailable ({repr(e)[:120]}) - skipped")

    def arc(cfg):
        out = []
        for ex in take(ds_load("allenai/ai2_arc", cfg, split="test"), n_per_task):
            ctx = "Question: " + ex["question"] + "\nAnswer:"
            conts = [" " + c for c in ex["choices"]["text"]]
            if ex["answerKey"] not in ex["choices"]["label"]:
                continue
            out.append((ctx, conts, ex["choices"]["label"].index(ex["answerKey"])))
        return out
    safe("arc_easy", lambda: arc("ARC-Easy"))
    safe("arc_challenge", lambda: arc("ARC-Challenge"))

    def hellaswag():
        out = []
        def pre(t):
            t = t.strip().replace(" [title]", ". ")
            t = re.sub("\\[.*?\\]", "", t)
            return t.replace("  ", " ")
        for ex in take(ds_load("Rowan/hellaswag", split="validation"), n_per_task):
            ctx = pre(ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize())
            out.append((ctx, [" " + pre(e) for e in ex["endings"]], int(ex["label"])))
        return out
    safe("hellaswag", hellaswag)

    def boolq():
        out = []
        for ex in take(ds_load("google/boolq", split="validation"), n_per_task):
            ctx = ex["passage"] + "\nQuestion: " + ex["question"] + "?\nAnswer:"
            out.append((ctx, [" no", " yes"], int(ex["answer"])))
        return out
    safe("boolq", boolq)

    def winogrande():
        out = []
        try:
            ds = ds_load("allenai/winogrande", "winogrande_xl", split="validation")
        except Exception:
            ds = ds_load("allenai/winogrande", "winogrande_debiased", split="validation")
        for ex in take(ds, n_per_task):
            s = ex["sentence"]; i = s.index("_")
            suffix = " " + s[i + 1:].strip()
            ctxs = [s[:i] + ex["option1"], s[:i] + ex["option2"]]
            out.append((ctxs, [suffix, suffix], int(ex["answer"]) - 1))     # partial evaluation
        return out
    safe("winogrande", winogrande)

    def piqa():
        last = None
        for name in ["ybisk/piqa", "baber/piqa", "nthngdy/piqa"]:
            try:
                ds = ds_load(name, split="validation"); break
            except Exception as e:
                last = e; ds = None
        if ds is None:
            raise last
        out = []
        for ex in take(ds, n_per_task):
            out.append(("Question: " + ex["goal"] + "\nAnswer:", [" " + ex["sol1"], " " + ex["sol2"]], int(ex["label"])))
        return out
    safe("piqa", piqa)

    def mmlu():
        out = []
        for ex in take(ds_load("cais/mmlu", "all", split="test"), n_per_task):
            letters = ["A", "B", "C", "D"]
            q = ex["question"].strip() + "\n" + "\n".join(f"{l}. {c}" for l, c in zip(letters, ex["choices"])) + "\nAnswer:"
            out.append((q, [" A", " B", " C", " D"], int(ex["answer"])))
        return out
    safe("mmlu", mmlu)
    return tasks


@torch.no_grad()
def score_choices(model, tok, ctx, conts):
    """Sum of continuation log-probs for each choice (ctx may be a list per choice: partial scoring)."""
    out = []
    for ci, c in enumerate(conts):
        cx = ctx[ci] if isinstance(ctx, list) else ctx
        a = tok(cx).input_ids; bb = tok(c, add_special_tokens=False).input_ids
        ids = torch.tensor([(a + bb)[-1024:]], device=DEV)
        nb = min(len(bb), ids.shape[1] - 1)
        logits = model(ids).logits[0, -nb - 1:-1].float()
        lp = torch.log_softmax(logits, -1).gather(1, ids[0, -nb:].unsqueeze(1)).sum().item()
        out.append(lp)
    return out


def eval_tasks(model, tok, tasks, limit=None):
    res = {}
    for name, exs in tasks.items():
        exs = exs if limit is None else exs[:limit]
        acc = accn = 0; preds = []
        for ctx, conts, gold in exs:
            lps = score_choices(model, tok, ctx, conts)
            p = max(range(len(lps)), key=lambda i: lps[i])
            pn = max(range(len(lps)), key=lambda i: lps[i] / max(1, len(conts[i])))
            acc += int(p == gold); accn += int(pn == gold); preds.append([round(x, 6) for x in lps])
        res[name] = {"acc": acc / max(1, len(exs)), "acc_norm": accn / max(1, len(exs)), "n": len(exs), "logprobs": preds}
    return res


# ============================================================ stages
def stage_timing(args):
    out = {}
    if DEV != "cuda":
        return out
    N, k, T = 8, 4, 2
    shapes = {"Llama-3-8B gate/up": (4096, 14336), "Llama-3-8B down": (14336, 4096), "Qwen2.5-7B up": (3584, 18944)}
    def bench(fn, reps=10):
        for _ in range(2): fn()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(reps): fn()
        torch.cuda.synchronize(); return (time.perf_counter() - t) / reps * 1e3
    for m in ([2048] if QUICK else [1024, 2048, 4096, 8192]):
        for nm, (d, n) in shapes.items():
            key = f"{nm} m={m}"
            try:
                X = torch.randn(m, d, device=DEV, dtype=torch.bfloat16)
                W = torch.randn(d, n, device=DEV, dtype=torch.bfloat16) * 0.02
                Wq, sw = quant_cols(W.float()); Wq = Wq.to(torch.int8).t().contiguous().t()
                code = RNSCode(N, k, T); code.check_exact = False
                t_local = bench(lambda: X @ W)
                t_w8a8 = bench(lambda: int8_gemm(quant_rows(X, 8)[0].to(torch.int8), Wq))
                t_coded_total = bench(lambda: code.linear(X, Wq, sw, None), reps=3)
                # one worker's GEMM: r=4 residues of an (m/k) x d share
                Sw = torch.randint(-125, 126, (4 * (m // k), d), device=DEV, dtype=torch.int8)
                t_worker = bench(lambda: torch.remainder(torch._int_mm(Sw, Wq), 251))
                N_gemm = N  # workers emulated serially on this GPU
                out[key] = {"local_bf16_ms": t_local, "local_w8a8_ms": t_w8a8, "one_worker_ms": t_worker,
                            "worker_vs_local_bf16": t_worker / t_local,
                            "coded_total_serial_ms": t_coded_total,
                            "coordinator_overhead_ms_est": max(0.0, t_coded_total - N_gemm * t_worker),
                            "coordinator_overhead_vs_local_bf16": max(0.0, t_coded_total - N_gemm * t_worker) / t_local}
                log("timing", key, {a: round(b, 3) for a, b in out[key].items()})
                del X, W, Wq, Sw; torch.cuda.empty_cache()
            except Exception as e:
                out[key] = {"error": repr(e)[:300]}; log("timing error", key, repr(e)[:200]); torch.cuda.empty_cache()
    return out


def stage_stress(args):
    out = []
    configs = [(8, 4, 2), (8, 2, 2), (6, 2, 1), (12, 4, 3), (16, 8, 4), (12, 6, 2)]
    shapes = [(256, 1536, 8960), (1024, 2048, 2048), (37, 896, 4864)] if not QUICK else [(64, 128, 256)]
    rng = random.Random(1)
    for (N, k, T) in configs:
        red = N - k - T
        for (m, d, n) in shapes:
            for trial in range(3 if not QUICK else 1):
                X = torch.randn(m, d, device=DEV) * (1 + 10 * rng.random())
                W = torch.randn(d, n, device=DEV) * 0.03
                Wq, sw = quant_cols(W); Wq = Wq.to(torch.int8).t().contiguous().t()
                for scenario in ["clean", "stragglers", "byz_correct", "byz_detect"]:
                    code = RNSCode(N, k, T, seed=trial)
                    if scenario == "stragglers" and red >= 1:
                        code.stragglers = set(rng.sample(range(N), red))
                    elif scenario == "byz_correct" and red >= 2:
                        code.byzantine = set(rng.sample(range(N), red // 2))
                    elif scenario == "byz_detect" and red >= 1:
                        code.byzantine = set(rng.sample(range(N), red))
                    elif scenario != "clean":
                        continue
                    rec = {"N": N, "k": k, "T": T, "shape": [m, d, n], "scenario": scenario,
                           "stragglers": sorted(code.stragglers), "byzantine": sorted(code.byzantine)}
                    try:
                        code.linear(X, Wq, sw, None)
                        rec.update(result="ok", exact=code.stats["exact"] == 1, flagged=code.stats["flagged"])
                    except RuntimeError as e:
                        rec.update(result="aborted", msg=str(e)[:80])
                    out.append(rec)
        log("stress", (N, k, T), "done;", sum(r.get("exact", False) for r in out), "exact /", len(out), "records so far")
    # two-sided product (Theorem 4) exactness
    tw = []
    for (N, k, T) in [(8, 2, 2), (8, 1, 2), (8, 3, 1), (12, 4, 2)]:
        for (m, n, d) in ([(512, 512, 128), (100, 300, 64)] if not QUICK else [(40, 30, 16)]):
            A = torch.randint(-127, 128, (m, d), device=DEV).float(); B = torch.randint(-127, 128, (n, d), device=DEV).float()
            code = RNSCode(N, k, T, seed=3)
            C = code.private_product(A, B, 127 * 127 * d)
            ref = (A.double() @ B.double().t()).round().to(torch.int64)
            ok = torch.equal(C, ref)
            code2 = RNSCode(N, k, T, seed=4); code2.byzantine = {0}
            red = N - (2 * (k + T) - 1)
            try:
                C2 = code2.private_product(A, B, 127 * 127 * d); byz = "corrected" if torch.equal(C2, ref) else "WRONG"
            except RuntimeError:
                byz = "detected/aborted"
            tw.append({"N": N, "k": k, "T": T, "m": m, "n": n, "d": d, "threshold": 2 * (k + T) - 1, "exact": ok,
                       "one_byzantine": byz, "redundancy": red})
            log("two-sided", tw[-1])
    return {"one_sided": out, "two_sided": tw,
            "summary": {"records": len(out), "ok": sum(r["result"] == "ok" for r in out),
                        "exact_ok": sum(r.get("exact", False) for r in out),
                        "aborted": sum(r["result"] == "aborted" for r in out)}}


def stage_tasks(args):
    res = {}
    tasks = get_tasks(args.task_n)
    for name in args.task_models:
        try:
            tok, model, state = load_model(name)
            r = res.setdefault(name, {})
            for tag, mode, ab, lim in [("bf16", "plain", 8, None), ("w8a8", "wq", 8, None), ("w8a16", "wq", 16, args.w8a16_task_n)]:
                set_mode(state, mode, abits=ab); t = time.time()
                r[tag] = eval_tasks(model, tok, tasks, limit=lim)
                log(name, tag, {t_: round(v["acc_norm"], 4) for t_, v in r[tag].items()}, f"{time.time()-t:.0f}s")
                jdump(res, "results2_tasks.partial.json")
            # coded: subset, must reproduce W8A8 log-probs exactly
            code = RNSCode(8, 4, 2, abits=8, seed=11)
            set_mode(state, "rns", code=code); t = time.time()
            rc = eval_tasks(model, tok, tasks, limit=args.coded_task_n)
            same = {tn: all(a == b for a, b in zip(rc[tn]["logprobs"], r["w8a8"][tn]["logprobs"][:len(rc[tn]["logprobs"])])) for tn in rc}
            r["rns_w8a8_subset"] = {tn: {"acc": rc[tn]["acc"], "acc_norm": rc[tn]["acc_norm"], "n": rc[tn]["n"],
                                         "logprobs_identical_to_w8a8": same[tn]} for tn in rc}
            r["rns_stats"] = dict(code.stats)
            log(name, "rns subset identical:", same, code.stats, f"{time.time()-t:.0f}s")
            for tg in ("bf16", "w8a8", "w8a16"):
                for tn in r[tg]:
                    r[tg][tn].pop("logprobs", None)
            jdump(res, "results2_tasks.partial.json")
            del model; torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            res[name] = {"error": "out of GPU memory"}; log("OOM", name); torch.cuda.empty_cache()
        except Exception:
            res[name] = {"error": traceback.format_exc()[-1500:]}; log("ERROR", name, traceback.format_exc()[-400:])
    return res


def stage_attention(args):
    res = {}
    for name in args.attn_models:
        try:
            tok, model, state = load_model(name)
            wins = wikitext_windows(tok, args.attn_windows, args.attn_seq)
            r = res.setdefault(name, {"windows": len(wins), "seq": args.attn_seq})
            def ppl():
                nll = cnt = 0
                for w in wins:
                    ids = w.unsqueeze(0).to(DEV)
                    with torch.no_grad():
                        lg = model(ids).logits[0, :-1].float()
                    v = F.cross_entropy(lg, ids[0, 1:], reduction="sum")
                    nll += v.item(); cnt += ids.shape[1] - 1
                return math.exp(nll / cnt)
            for tag, mode, attn in [("bf16", "plain", "plain"), ("w8a8_linear", "wq", "plain"),
                                    ("w8a8_linear+int8_attn", "wq", "qattn")]:
                set_mode(state, mode); state["attn"] = attn; t = time.time()
                r[tag] = {"ppl": ppl(), "sec": time.time() - t}; log(name, tag, r[tag])
            # fully coded: linear (Thm 2) + attention (Thm 4), N=8, k=1, T=2 for attention (threshold 7)
            code = RNSCode(8, 4, 2, abits=8, seed=21)
            acode = RNSCode(8, 1, 2, abits=8, seed=22)
            set_mode(state, "rns", code=code); state.update(attn="cattn", acode=acode, check=True, att_checked=0, att_exact=0)
            t = time.time()
            r["fully_coded"] = {"ppl": ppl(), "sec": time.time() - t, "linear_exact": f"{code.stats['exact']}/{code.stats['gemms']}",
                                "attention_products_exact": f"{state['att_exact']}/{state['att_checked']}"}
            log(name, "fully_coded", r["fully_coded"])
            state.update(attn="plain", check=False)
            del model; torch.cuda.empty_cache()
        except Exception:
            res[name] = {"error": traceback.format_exc()[-1500:]}; log("ERROR attention", name, traceback.format_exc()[-400:])
            torch.cuda.empty_cache()
    return res


def stage_generate(args):
    res = {}
    try:
        ds = ds_load("openai/gsm8k", "main", split="test")
        train = ds_load("openai/gsm8k", "main", split="train")
    except Exception as e:
        log("GSM8K unavailable:", repr(e)[:150]); return {"error": "gsm8k unavailable"}
    shots = "".join(f"Question: {train[i]['question']}\nAnswer: {train[i]['answer']}\n\n" for i in range(4))
    probs = [ds[i] for i in range(args.gen_n)]

    def extract(t):
        m = re.findall(r"####\s*(-?[\d,\.]+)", t)
        if not m:
            m = re.findall(r"(-?\d[\d,]*\.?\d*)", t)
        return m[-1].replace(",", "").rstrip(".") if m else None

    for name in args.gen_models:
        try:
            tok, model, state = load_model(name)
            r = res.setdefault(name, {})
            @torch.no_grad()
            def greedy(prompt, max_new):
                ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
                out = model(ids, use_cache=True)
                past = out.past_key_values; nxt = out.logits[0, -1].argmax().view(1, 1); gen = [nxt.item()]
                for _ in range(max_new - 1):
                    o = model(nxt, past_key_values=past, use_cache=True)
                    past = o.past_key_values; nxt = o.logits[0, -1].argmax().view(1, 1); gen.append(nxt.item())
                    txt = tok.decode(gen)
                    if "\n\nQuestion" in txt or "Question:" in txt[-12:] or nxt.item() == tok.eos_token_id:
                        break
                return gen
            outs = {}
            for tag, mode, ab, n in [("bf16", "plain", 8, args.gen_n), ("w8a8", "wq", 8, args.gen_n),
                                     ("rns_w8a8", "rns", 8, args.gen_coded_n), ("w8a16", "wq", 16, min(args.gen_n, args.gen_w8a16_n))]:
                code = RNSCode(8, 4, 2, abits=ab, seed=31) if mode == "rns" else None
                set_mode(state, mode, code=code, abits=ab); t = time.time()
                correct = 0; gens = []
                for ex in probs[:n]:
                    g = greedy(shots + f"Question: {ex['question']}\nAnswer:", args.gen_max)
                    gens.append(g)
                    correct += int(extract(tok.decode(g)) == extract(ex["answer"]))
                outs[tag] = gens
                r[tag] = {"acc": correct / n, "n": n, "sec": time.time() - t,
                          "gen_tokens": sum(len(g) for g in gens)}
                if code is not None:
                    r[tag]["stats"] = dict(code.stats)
                    r[tag]["token_identical_to_w8a8"] = sum(int(a == b) for a, b in zip(gens, outs["w8a8"]))
                log(name, tag, r[tag])
                jdump(res, "results2_generate.partial.json")
            del model; torch.cuda.empty_cache()
        except Exception:
            res[name] = {"error": traceback.format_exc()[-1500:]}; log("ERROR generate", name, traceback.format_exc()[-400:])
            torch.cuda.empty_cache()
    return res


def make_real_code(N=8, k=4, T=2, seed=0, iters=400, restarts=2):
    g = torch.Generator().manual_seed(seed)
    def metric(E):
        amp = max(torch.linalg.matrix_norm(torch.linalg.pinv(E[list(S)])[:k], 2).item() for S in itertools.combinations(range(N), k + T))
        gain = max(torch.linalg.matrix_norm(torch.linalg.solve(E[list(S), k:], E[list(S), :k]), 2).item() ** 2 for S in itertools.combinations(range(N), T))
        return amp * math.sqrt(gain)
    best, bv = None, float("inf")
    for _ in range(restarts):
        E = torch.randn(N, k + T, generator=g, dtype=torch.float64); E /= E.norm(dim=1, keepdim=True); v = metric(E); step = 0.3
        for it in range(iters):
            Fm = E + step * torch.randn(E.shape, generator=g, dtype=torch.float64); Fm /= Fm.norm(dim=1, keepdim=True)
            v2 = metric(Fm)
            if v2 < v: E, v = Fm, v2
            elif it % 100 == 99: step *= 0.7
        if v < bv: best, bv = E, v
    return best


def stage_privacy(args):
    """Neural inversion of prompt tokens from what a coalition sees, incl. canary PINs."""
    name = args.priv_model
    tok, model, state = load_model(name)
    E = make_real_code(iters=60 if QUICK else 400).float().to(DEV)
    N, k, T = 8, 4, 2
    worst = max(itertools.combinations(range(N), T),
                key=lambda S: torch.linalg.matrix_norm(torch.linalg.solve(E[list(S), k:].double(), E[list(S), :k].double()), 2).item())
    worst = list(worst)
    nL = len(model.model.layers)
    layers = sorted({0, nL // 4, nL // 2, nL - 1})
    caps = {}
    hooks = []
    for li in layers:
        lin = model.model.layers[li].self_attn.q_proj
        def mk(li):
            def h(mod, inp):
                caps.setdefault(li, []).append(inp[0][0].float().cpu())
            return h
        hooks.append(lin.register_forward_pre_hook(mk(li)))
    L = 256
    tr_w = wikitext_windows(tok, 40 if QUICK else 400, L, split="train")
    te_w = wikitext_windows(tok, 8 if QUICK else 60, L, split="test")
    # canary prompts with random 6-digit PINs embedded in natural text
    rng = random.Random(5)
    tmpl = ["My bank PIN is {} and I keep it secret.", "The door code for the lab is {}, please do not share it.",
            "Patient record: account number {} was updated yesterday.", "Use verification code {} to log in to your account."]
    canaries, pins = [], []
    for i in range(16 if QUICK else 200):
        pin = "".join(str(rng.randrange(10)) for _ in range(6))
        filler = te_w[i % len(te_w)][:200].tolist()
        text_ids = tok(" " + tmpl[i % len(tmpl)].format(" ".join(pin)), add_special_tokens=False).input_ids
        ids = (filler + text_ids)[:L]
        ids = ids + filler[: L - len(ids)]
        pos = [j for j in range(len(filler), min(L, len(filler) + len(text_ids)))
               if len(tok.decode([ids[j]]).strip()) == 1 and tok.decode([ids[j]]).strip().isdigit()]
        canaries.append(torch.tensor(ids)); pins.append(pos)
    def run(ws):
        caps.clear(); state["mode"] = "plain"
        with torch.no_grad():
            for w in ws:
                model(w.unsqueeze(0).to(DEV))
        return {li: torch.stack(v) for li, v in caps.items()}
    Atr, Ate, Aca = run(tr_w), run(te_w), run(canaries)
    for h in hooks: h.remove()
    emb_cpu = model.get_input_embeddings().weight.detach().float().cpu()
    del model; torch.cuda.empty_cache()
    Ytr, Yte, Yca = torch.stack(tr_w), torch.stack(te_w), torch.stack(canaries)
    emb = emb_cpu                                                        # [V, d]
    b = L // k
    gen = torch.Generator(device=DEV); gen.manual_seed(7)

    def views(X, scheme, beta=None):
        """X: [P, L, d] (cpu) -> view rows [P, b, feat] for target block j (all blocks stacked as separate samples)."""
        X = X.to(DEV)
        P, _, d = X.shape
        blocks = X.view(P, k, b, d)
        if scheme == "uncoded":
            return blocks                                            # [P, k, b, d]: worker j sees block j
        if scheme == "analog":
            rms = X.pow(2).mean(1, keepdim=True).sqrt() + 1e-6        # per prompt, per channel
            R = torch.randn(P, T, b, d, generator=gen, device=DEV) * (beta * rms.unsqueeze(1))
            U = torch.cat([blocks, R], 1)                            # [P, k+T, b, d]
            S = torch.einsum("nc,pcbd->pnbd", E[worst], U)           # [P, T, b, d]
            v = S.permute(0, 2, 1, 3).reshape(P, b, T * d)
            return v.unsqueeze(1).expand(P, k, b, T * d)
        if scheme == "rns":
            Xq, _ = quant_rows(X, 8)
            p = 251
            blocks_q = torch.remainder(Xq.view(P, k, b, d), p)
            Mk = torch.randint(0, p, (P, T, b, d), generator=gen, device=DEV).float()
            U = torch.cat([blocks_q, Mk], 1)
            Vw = torch.tensor([[pow(a + 1, j, p) for j in range(k + T)] for a in worst], device=DEV, dtype=torch.float32)
            S = torch.remainder(torch.einsum("nc,pcbd->pnbd", Vw, U), p)
            S = torch.where(S > p // 2, S - p, S) / 125.0
            v = S.permute(0, 2, 1, 3).reshape(P, b, T * d)
            return v.unsqueeze(1).expand(P, k, b, T * d)

    def train_eval(li, scheme, beta=None):
        Vtr, Vte, Vca = views(Atr[li], scheme, beta), views(Ate[li], scheme, beta), views(Aca[li], scheme, beta)
        feat = Vtr.shape[-1]
        def flat(Vx, Y):
            P = Vx.shape[0]
            tokens = Y.view(P, k, b).to(DEV)
            onehot_block = torch.eye(k, device=DEV).view(1, k, 1, k).expand(P, k, b, k)
            return torch.cat([Vx, onehot_block], -1).reshape(-1, feat + k), tokens.reshape(-1)
        Xtr, ytr = flat(Vtr, Ytr); Xte, yte = flat(Vte, Yte); Xca, yca = flat(Vca, Yca)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-5
        net = nn.Sequential(nn.Linear(feat + k, 2048), nn.GELU(), nn.Linear(2048, 2048), nn.GELU(), nn.Linear(2048, emb.shape[1])).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
        E_n = emb.to(DEV)
        steps = 200 if QUICK else 3000
        bs = 1024
        for s in range(steps):
            idx = torch.randint(0, Xtr.shape[0], (bs,), device=DEV)
            h = net((Xtr[idx] - mu) / sd)
            loss = F.cross_entropy(h @ E_n.t(), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            if s == steps // 2:
                for g_ in opt.param_groups: g_["lr"] = 3e-4
        @torch.no_grad()
        def pred(Xx):
            return torch.cat([(net((Xx[i:i + 4096] - mu) / sd) @ E_n.t()).argmax(-1) for i in range(0, Xx.shape[0], 4096)])
        acc = (pred(Xte) == yte).float().mean().item()
        pc = pred(Xca).view(len(canaries), k, b)
        # canary digits: position j in prompt -> block j // b, row j % b
        dig_hits = full_hits = 0; ndig = 0
        for ci, pos in enumerate(pins):
            okall = True
            for j in pos:
                hit = int(pc[ci, j // b, j % b].item() == Yca[ci, j].item())
                dig_hits += hit; ndig += 1; okall &= bool(hit)
            full_hits += int(okall and len(pos) > 0)
        return {"token_acc": acc, "canary_digit_acc": dig_hits / max(1, ndig), "canary_full_pin": full_hits / len(pins),
                "final_loss": float(loss.item())}

    vals, cnts = torch.unique(Ytr, return_counts=True)
    res = {"model": name, "layers": layers, "coalition": worst, "N": N, "k": k, "T": T,
           "prior_most_frequent_token_acc": (Yte == vals[cnts.argmax()]).float().mean().item(),
           "prior_digit_acc": 0.1, "train_prompts": len(tr_w), "test_prompts": len(te_w), "canaries": len(canaries), "by_layer": {}}
    for li in layers:
        for scheme, beta in [("uncoded", None), ("analog", 0.0), ("analog", 1.0), ("analog", 10.0), ("analog", 100.0), ("rns", None)]:
            key = scheme if beta is None else f"analog_beta{beta:g}"
            t = time.time()
            try:
                res["by_layer"].setdefault(str(li), {})[key] = train_eval(li, scheme, beta)
            except Exception:
                res["by_layer"].setdefault(str(li), {})[key] = {"error": traceback.format_exc()[-600:]}
            log("privacy layer", li, key, res["by_layer"][str(li)][key], f"{time.time()-t:.0f}s")
            jdump(res, "results2_privacy.partial.json")
    return res


STAGES = {"timing": stage_timing, "stress": stage_stress, "tasks": stage_tasks, "attention": stage_attention,
          "generate": stage_generate, "privacy": stage_privacy}


def main():
    global QUICK
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=["timing", "stress", "attention", "generate", "privacy", "tasks"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run stages even if results exist")
    ap.add_argument("--task-models", nargs="+", default=["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "HuggingFaceTB/SmolLM2-1.7B"])
    ap.add_argument("--task-n", type=int, default=500)
    ap.add_argument("--w8a16-task-n", type=int, default=200)
    ap.add_argument("--coded-task-n", type=int, default=100)
    ap.add_argument("--attn-models", nargs="+", default=["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"])
    ap.add_argument("--attn-windows", type=int, default=8)
    ap.add_argument("--attn-seq", type=int, default=512)
    ap.add_argument("--gen-models", nargs="+", default=["Qwen/Qwen2.5-1.5B"])
    ap.add_argument("--gen-n", type=int, default=150)
    ap.add_argument("--gen-w8a16-n", type=int, default=80)
    ap.add_argument("--gen-coded-n", type=int, default=20)
    ap.add_argument("--gen-max", type=int, default=256)
    ap.add_argument("--priv-model", default="Qwen/Qwen2.5-1.5B")
    args = ap.parse_args()
    QUICK = args.quick
    prefix = "quick_" if QUICK else ""
    if QUICK:
        args.task_models = ["Qwen/Qwen2.5-0.5B"]; args.task_n = 8; args.coded_task_n = 3
        args.attn_models = ["Qwen/Qwen2.5-0.5B"]; args.attn_windows = 1; args.attn_seq = 64
        args.gen_models = ["Qwen/Qwen2.5-0.5B"]; args.gen_n = 2; args.gen_coded_n = 1; args.gen_max = 24
        args.priv_model = "Qwen/Qwen2.5-0.5B"
    env = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
           "python": sys.version.split()[0], "platform": platform.platform(), "args": vars(args)}
    try:
        import transformers, datasets
        env["transformers"] = transformers.__version__; env["datasets"] = datasets.__version__
    except Exception:
        pass
    log("environment", env)
    for st in args.stages:
        path = f"{prefix}results2_{st}.json"
        if os.path.exists(path) and not args.force:
            log(f"stage {st}: {path} exists, skipping (use --force to redo)"); continue
        log(f"===== stage {st} =====")
        t = time.time()
        try:
            r = STAGES[st](args)
            r = {"result": r, "env": env, "seconds": time.time() - t}
            jdump(r, path)
            log(f"stage {st} finished in {time.time()-t:.0f}s -> {path}")
        except Exception:
            log(f"stage {st} FAILED:", traceback.format_exc()[-1500:])
            jdump({"error": traceback.format_exc()[-3000:], "env": env}, f"{prefix}results2_{st}.FAILED.json")
        if DEV == "cuda":
            torch.cuda.empty_cache()
    log("ALL DONE")


if __name__ == "__main__":
    main()
