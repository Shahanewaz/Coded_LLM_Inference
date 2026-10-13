"""Check the weight-split polynomial code (Prop. 2): X split into k row blocks (+T masks),
W split into l column blocks encoded publicly; recovery threshold (k+T)*l; exact via RNS."""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from coded import quant_act, quant_weight, primes_for, centered, mat_inv_mod, crt_centered

def run(N, k, T, l, m=48, d=96, h=60, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((m, d)); W = rng.standard_normal((d, h)) * 0.1
    Xq, sx = quant_act(X); Wq, sw = quant_weight(W)
    primes = primes_for(127 * 127 * d)
    b, hb = m // k, h // l
    KT = k + T
    res = []
    for p in primes:
        a = np.arange(1, N + 1)
        U = np.concatenate([np.mod(Xq.reshape(k, b, d), p).astype(np.int64), rng.integers(0, p, (T, b, d))])
        Vx = np.array([[pow(int(x), j, p) for j in range(KT)] for x in a])
        Vw = np.array([[pow(int(x), KT * j, p) for j in range(l)] for x in a])
        Wb = np.mod(Wq.reshape(d, l, hb).transpose(1, 0, 2), p).astype(np.int64)
        S = centered(np.tensordot(Vx, U, (1, 0)) % p, p)                 # shares (int8 range)
        CW = centered(np.tensordot(Vw, Wb, (1, 0)) % p, p)               # coded public weights per worker (int8 range)
        assert np.abs(S).max() <= 125 and np.abs(CW).max() <= 125
        Z = np.stack([S[i].astype(np.float64) @ CW[i] for i in range(N)]).astype(np.int64) % p
        use = list(rng.choice(N, KT * l, replace=False))                # any (k+T)l responders
        V = np.array([[pow(int(a[i]), e, p) for e in range(KT * l)] for i in use])
        coef = np.tensordot(mat_inv_mod(V, p), Z[use], (1, 0)) % p      # coefficient of z^(j + KT*mm)
        out = np.zeros((k, b, l, hb), dtype=np.int64)
        for j in range(k):
            for mm in range(l):
                out[j, :, mm, :] = coef[j + KT * mm]
        res.append(out.reshape(m, h))
    Yint = crt_centered(res, primes)
    return np.array_equal(Yint, (Xq @ Wq).astype(np.int64))

for N, k, T, l in [(8, 2, 2, 2), (12, 2, 1, 4), (16, 3, 1, 4), (9, 2, 1, 3)]:
    print((N, k, T, l), "threshold", (k + T) * l, "exact:", run(N, k, T, l))
