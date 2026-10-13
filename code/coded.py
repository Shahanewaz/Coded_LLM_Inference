"""Coded execution of public-weight linear layers  Y = X W + b  with private X.

Two families are implemented:

* FFCode      -- "quantize the secret, not the share": X is quantized to integers at the
                 trusted coordinator, then secret-shared with a polynomial (Shamir/LCC-type)
                 code over each prime field of a residue number system (RNS) whose primes are
                 < 256, so every share residue and every weight is an int8 and each worker GEMM
                 is an int8 x int8 -> int32 product.  Decoding is exact; any T colluding workers
                 see uniformly random residues; Reed-Solomon redundancy detects/corrects
                 Byzantine responses exactly.
* AnalogCode  -- real-valued coded computing with Gaussian masks (the analog baseline), executed
                 with emulated BF16 / FP16 / FP32 tensor-core semantics.

Worker arithmetic is emulated bit-exactly: int8 GEMMs with int32 accumulation are computed in
float64 (exact because every partial sum is an integer < 2^53); BF16/FP16 GEMMs round their
inputs to the format, accumulate in FP32 and round the output to the format.
"""
import itertools
import numpy as np

SMALL_PRIMES = [251, 241, 239, 233, 229, 227, 223, 211, 199, 197, 193, 191, 181, 179, 173, 167]   # all < 2^8: centered residues in [-125, 125] fit int8


# ============================================================ quantization (coordinator side)
def quant_act(X, bits=8):
    qmax = 2 ** (bits - 1) - 1
    s = np.abs(X).max(axis=1, keepdims=True).astype(np.float64) / qmax
    s[s == 0] = 1.0
    Xq = np.clip(np.rint(X / s), -qmax, qmax)
    return Xq, s[:, 0]


def quant_weight(W, bits=8):
    qmax = 2 ** (bits - 1) - 1
    s = np.abs(W).max(axis=0).astype(np.float64) / qmax
    s[s == 0] = 1.0
    Wq = np.clip(np.rint(W / s), -qmax, qmax)
    return Wq, s


class QuantCache:
    """Per-output-channel int weights are public and computed once."""
    def __init__(self, wbits=8):
        self.wbits, self.c = wbits, {}

    def get(self, name, W):
        if name not in self.c:
            self.c[name] = quant_weight(W.astype(np.float64), self.wbits)
        return self.c[name]


def int_reference_linear(abits=8, wbits=8):
    """Uncoded W{wbits}A{abits} reference: Y = s_x s_w (Xq @ Wq) + b, integer product exact."""
    qc = QuantCache(wbits)

    def linear(name, X, W, b):
        Wq, sw = qc.get(name, W)
        Xq, sx = quant_act(X.astype(np.float64), abits)
        Yint = Xq @ Wq                                           # exact (|.| < 2^53)
        return (Yint * sx[:, None] * sw[None, :] + b).astype(np.float32)
    linear.qc = qc
    return linear


# ============================================================ modular helpers
def inv_mod(a, p):
    return pow(int(a) % p, p - 2, p)


def mat_inv_mod(A, p):
    """Inverse of a small square integer matrix over F_p (Gauss-Jordan)."""
    n = A.shape[0]
    M = np.concatenate([np.array(A, dtype=np.int64) % p, np.eye(n, dtype=np.int64)], axis=1)
    for c in range(n):
        piv = next(r for r in range(c, n) if M[r, c] % p)
        M[[c, piv]] = M[[piv, c]]
        M[c] = (M[c] * inv_mod(M[c, c], p)) % p
        for r in range(n):
            if r != c and M[r, c]:
                M[r] = (M[r] - M[r, c] * M[c]) % p
    return M[:, n:]


def centered(x, p):
    x = np.mod(x, p)
    return np.where(x > p // 2, x - p, x)


def primes_for(bound):
    """Smallest prefix of SMALL_PRIMES whose product exceeds 2*bound (signed range)."""
    prod, out = 1, []
    for p in SMALL_PRIMES:
        out.append(p); prod *= p
        if prod > 2 * bound:
            return out
    raise ValueError("not enough small primes")


def crt_centered(res, primes):
    """res: list of int64 arrays (residues mod p_i) -> centered integers in (-M/2, M/2]."""
    M = 1
    for p in primes:
        M *= p
    x = np.zeros_like(res[0], dtype=np.int64)
    for r, p in zip(res, primes):
        Mi = M // p
        coef = (Mi * inv_mod(Mi % p, p)) % M        # < M < 2^63/128
        x = (x + (np.mod(r, p).astype(np.int64) * coef) % M) % M
    return np.where(x > M // 2, x - M, x)


# ============================================================ finite-field (RNS) secret-shared code
class FFCode:
    """Polynomial (Shamir-packed / LCC-type) code over each F_p, p in an RNS basis.

    Share for worker i (evaluation point a_i):  S_i = sum_{j<k} X_j a_i^j + sum_{t<T} R_t a_i^{k+t}  (mod p)
    Worker returns Z_i = S_i W_q (mod p), the evaluation of a degree-(k+T-1) polynomial whose first
    k coefficients are X_j W_q.  Any k+T responses decode; any T shares are uniform.
    """

    def __init__(self, N, k, T, abits=8, wbits=8, seed=0):
        assert N >= k + T
        self.N, self.k, self.T, self.abits, self.wbits = N, k, T, abits, wbits
        self.alphas = np.arange(1, N + 1, dtype=np.int64)
        self.rng = np.random.default_rng(seed)
        self.qc = QuantCache(wbits)
        self.byzantine = {}        # worker -> callable(name, prime_index, Z, ctx) -> Z'
        self.stragglers = set()
        self.log = []              # per-op records (detections etc.)
        self.check_exact = False
        self.verify = True         # Reed-Solomon consistency check before decoding
        self.decode_set = None     # last decoding set used (public to an attacker who can predict it)
        self.capture = None        # optional callback(name, shares_by_prime) for attack studies

    def vander(self, p, rows=None):
        rows = range(self.N) if rows is None else rows
        a = self.alphas[list(rows)]
        return np.stack([np.array([pow(int(x), j, p) for j in range(self.k + self.T)]) for x in a]).astype(np.int64)

    # ---- the protocol for one linear layer
    def linear(self, name, X, W, b):
        k, T, N = self.k, self.T, self.N
        Wq, sw = self.qc.get(name, W)
        Xq, sx = quant_act(X.astype(np.float64), self.abits)
        m, din = Xq.shape
        bound = (2 ** (self.abits - 1) - 1) * (2 ** (self.wbits - 1) - 1) * din
        primes = primes_for(bound)
        bsz = -(-m // k)
        Xp = np.zeros((k * bsz, din), dtype=np.int64)
        Xp[:m] = Xq.astype(np.int64)
        blocks = Xp.reshape(k, bsz, din)

        # ---- encode (coordinator): shares[pi][i] = int8-ranged residues
        shares = []
        for p in primes:
            U = np.concatenate([np.mod(blocks, p), self.rng.integers(0, p, size=(T, bsz, din))], axis=0)
            V = self.vander(p)
            S = np.tensordot(V, U, axes=(1, 0)) % p          # [N, bsz, din]
            shares.append(centered(S, p))
        if self.capture is not None:
            self.capture(name, shares, primes)

        # ---- workers: one int8 x int8 -> int32 GEMM per residue (batched here for speed)
        stack = np.stack(shares, 0).reshape(-1, din).astype(np.float64)    # [r*N*bsz, din]
        Zall = (stack @ Wq)                                                  # exact integers
        assert np.abs(Zall).max() < 2 ** 31, "int32 accumulator overflow"
        Zall = Zall.reshape(len(primes), N, bsz, -1).astype(np.int64)
        Z = []
        for pi, p in enumerate(primes):
            Zp = centered(Zall[pi], p)                       # worker reduces and returns int8 residues
            for i, fn in self.byzantine.items():
                Zp[i] = fn(name, pi, p, Zp[i].copy(), self)
            Z.append(Zp)

        # ---- decode (coordinator): RS error detection / correction, then MDS solve and CRT
        avail = [i for i in range(N) if i not in self.stragglers]
        rec = {"name": name, "detected": False, "excluded": []}
        res = []
        bad = self._locate_errors(Z, primes, avail) if self.verify else []
        if bad:
            rec["detected"] = True; rec["excluded"] = bad
            if bad == [-1]:
                self.log.append(rec)
                raise RuntimeError("Byzantine responses detected but not correctable: abort")
        use = [i for i in avail if i not in bad][: k + T]
        self.decode_set = use
        if len(use) < k + T:
            raise RuntimeError("not enough clean responses")
        for pi, p in enumerate(primes):
            Vinv = mat_inv_mod(self.vander(p, use), p)
            Zu = np.mod(Z[pi][use], p)
            coeff = np.tensordot(Vinv[:k], Zu, axes=(1, 0)) % p   # [k, bsz, dout]  = X_j W_q mod p
            res.append(coeff)
        Yint = crt_centered(res, primes).reshape(k * bsz, -1)[:m].astype(np.float64)
        if self.check_exact:
            rec["exact"] = bool(np.array_equal(Yint, Xq @ Wq))
        rec["primes"] = len(primes)
        self.log.append(rec)
        return (Yint * sx[:, None] * sw[None, :] + b).astype(np.float32)

    def _consistent(self, Z, primes, rows):
        """True if responses on `rows` all lie on one degree-(k+T-1) polynomial (every prime)."""
        kt = self.k + self.T
        base, rest = rows[:kt], rows[kt:]
        if not rest:
            return True
        for pi, p in enumerate(primes):
            coef = np.tensordot(mat_inv_mod(self.vander(p, base), p), np.mod(Z[pi][base], p), axes=(1, 0)) % p
            pred = np.tensordot(self.vander(p, rest), coef, axes=(1, 0)) % p
            if not np.array_equal(pred, np.mod(Z[pi][rest], p)):
                return False
        return True

    def _locate_errors(self, Z, primes, avail):
        """Exact syndrome check; if it fails, find the smallest set of workers whose removal
        makes the rest consistent (unique when 2|bad| <= |avail| - (k+T))."""
        if self._consistent(Z, primes, avail):
            return []
        red = len(avail) - (self.k + self.T)
        for a in range(1, red // 2 + 1):
            for bad in itertools.combinations(avail, a):
                rest = [i for i in avail if i not in bad]
                if self._consistent(Z, primes, rest):
                    return list(bad)
        return [-1]   # detected but not correctable -> caller aborts / falls back


# ============================================================ analog (real-valued) baseline
def to_bf16(x):
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32)


def to_fmt(x, fmt):
    if fmt == "fp32":
        return np.asarray(x, dtype=np.float32)
    if fmt == "bf16":
        return to_bf16(x)
    if fmt == "fp16":
        with np.errstate(over="ignore"):
            return np.asarray(x, dtype=np.float16).astype(np.float32)
    if fmt == "fp64":
        return np.asarray(x, dtype=np.float64)
    raise ValueError(fmt)


def fmt_gemm(A, W, fmt):
    """Tensor-core semantics: inputs in fmt, FP32 accumulate (FP64 for fp64), output rounded to fmt."""
    if fmt == "fp64":
        return A.astype(np.float64) @ W.astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        Y = to_fmt(A, fmt).astype(np.float32) @ to_fmt(W, fmt).astype(np.float32)
    return to_fmt(Y, fmt)


def plain_fmt_linear(fmt):
    def linear(name, X, W, b):
        return (fmt_gemm(X, W, fmt).astype(np.float64) + b).astype(np.float32)
    return linear


def make_real_code(N, k, T, seed=0, draws=200):
    """Row-normalized Gaussian code E=[C B], screened for the worst decoder amplification over all
    (k+T)-subsets (straggler resilience) -- the same construction as the original draft."""
    rng = np.random.default_rng(seed)
    best, bestv = None, np.inf
    for _ in range(draws):
        E = rng.standard_normal((N, k + T))
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        worst = max(np.linalg.norm(np.linalg.pinv(E[list(S)])[:k], 2)
                    for S in itertools.combinations(range(N), k + T))
        if worst < bestv:
            best, bestv = E, worst
    return best


def gaussian_leakage_per_entry(E, k, T, beta):
    """Worst T-coalition Gaussian-surrogate leakage (nats) per (row position, channel):
    0.5 logdet(I + beta^-2 A A^T), A = B_C^{-1} C_C, for x ~ N(0, rho I), masks ~ N(0, beta^2 rho)."""
    N = E.shape[0]
    C, B = E[:, :k], E[:, k:]
    worst = 0.0
    for S in itertools.combinations(range(N), T):
        S = list(S)
        A = np.linalg.solve(B[S], C[S])
        val = 0.5 * np.linalg.slogdet(np.eye(T) + (A @ A.T) / beta ** 2)[1]
        worst = max(worst, val)
    return worst


class AnalogCode:
    def __init__(self, N, k, T, fmt="bf16", beta=10.0, seed=0, E=None):
        self.N, self.k, self.T, self.fmt, self.beta = N, k, T, fmt, beta
        self.E = make_real_code(N, k, T, seed) if E is None else E
        self.rng = np.random.default_rng(seed + 1)
        self.byzantine = {}
        self.capture = None
        self.overflow = False
        self.log = []

    def linear(self, name, X, W, b):
        k, T, N, E = self.k, self.T, self.N, self.E
        X = X.astype(np.float64)
        m, din = X.shape
        bsz = -(-m // k)
        Xp = np.zeros((k * bsz, din)); Xp[:m] = X
        blocks = Xp.reshape(k, bsz, din)
        rms = np.sqrt((X ** 2).mean(axis=0)) + 1e-12                        # per-channel scale
        R = self.rng.standard_normal((T, bsz, din)) * (self.beta * rms)
        U = np.concatenate([blocks, R], 0)
        S = np.tensordot(E, U, axes=(1, 0))                                 # [N, bsz, din] (fp64)
        S = to_fmt(S, self.fmt)                                              # share stored in worker format
        if self.capture is not None:
            self.capture(name, S, rms)
        Z = fmt_gemm(S.reshape(-1, din), W, self.fmt).reshape(N, bsz, -1).astype(np.float64)
        for i, fn in self.byzantine.items():
            Z[i] = fn(name, Z[i].copy(), self)
        if not np.all(np.isfinite(Z)):
            self.overflow = True
        Dec = np.linalg.pinv(E)[:k]                                          # LS decoder, all N responses
        Y = np.tensordot(Dec, Z, axes=(1, 0)).reshape(k * bsz, -1)[:m]
        Zf = Z.reshape(N, -1)
        self.log.append({"name": name, "resid": float(np.linalg.norm(Zf - E @ (np.linalg.pinv(E) @ Zf))),
                         "znorm": float(np.linalg.norm(Zf))})
        return (Y + b).astype(np.float32)
