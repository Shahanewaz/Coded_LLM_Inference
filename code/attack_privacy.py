"""Prompt-token recovery by a coalition of T colluding workers.

For selected outsourced GEMMs we capture the private activation X (one prompt = m tokens), then
synthesize exactly what a coalition sees under each scheme:
  * uncoded token sharding  : worker sees one raw row-block X_j
  * analog coded (N,k,T)    : the T shares  E_C [X_1..X_k; R_1..R_T]  (Gaussian masks, sigma_c = beta*rms_c)
  * finite-field coded      : the T shares (RNS residues of the int8-quantized X plus uniform masks)
A learned attacker (ridge regression from the coalition's view-row to the GPT-2 input embedding of
the token in block j at that row, followed by nearest-neighbour decoding over the 50,257-token
vocabulary) is trained on public text (WikiText-2 train) and evaluated on held-out WikiText-2 test
prompts.  Top-1 token recovery is reported, averaged over the k blocks.
"""
import sys, os, json, itertools, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from gpt2np import GPT2, Tokenizer
from coded import quant_act, primes_for, centered, gaussian_leakage_per_entry

A = "/home/claude/assets"
OUT = "/home/claude/code/results"
N, K, T, M = 8, 4, 2, 256
OPS = ["h0.attn.c_attn", "h6.attn.c_attn", "h11.mlp.c_fc"]
BETAS = [0.0, 1.0, 10.0, 100.0, 1000.0]


def prompts(tok, path, n, seed):
    cache = path + ".ids.npy"
    if not os.path.exists(cache):
        np.save(cache, np.array(tok.encode(open(path).read()[:1500000]), dtype=np.int64))
    ids = np.load(cache)
    rng = np.random.default_rng(seed)
    starts = rng.choice(len(ids) - M, n, replace=False)
    return [ids[s:s + M] for s in starts]


def capture(g, ps):
    acts = {op: [] for op in OPS}
    def hook(name, X):
        if name in acts: acts[name].append(X.astype(np.float64).copy())
    last = max(int(o.split(".")[0][1:]) for o in OPS) + 1
    for p in ps:
        g.forward(p, hook=hook, upto_layer=last)
    return acts


def views(X, scheme, rng, E=None, coalition=None, beta=None):
    """Return view rows [b, feat] for one prompt and the block-structured token index map."""
    m, d = X.shape; b = m // K
    blocks = X.reshape(K, b, d)
    if scheme == "uncoded":
        return blocks                                    # [K, b, d]: worker j sees block j
    if scheme == "analog":
        rms = np.sqrt((X ** 2).mean(0)) + 1e-12
        R = rng.standard_normal((T, b, d)) * (beta * rms)
        U = np.concatenate([blocks, R], 0)
        S = np.tensordot(E[coalition], U, axes=(1, 0))   # [T, b, d]
        return np.concatenate(list(S), axis=1).astype(np.float32)   # [b, T*d]
    if scheme == "ff":
        Xq, _ = quant_act(X, 8)
        qb = Xq.reshape(K, b, d).astype(np.int64)
        primes = primes_for(127 * 127 * d)[:1]   # residues of different primes are independent and uniform; one suffices
        feats = []
        for p in primes:
            U = np.concatenate([np.mod(qb, p), rng.integers(0, p, size=(T, b, d))], 0)
            a = np.array(coalition) + 1
            V = np.stack([[pow(int(x), j, p) for j in range(K + T)] for x in a])
            S = centered(np.tensordot(V, U, axes=(1, 0)) % p, p)
            feats += list(S)
        return np.concatenate(feats, axis=1).astype(np.float32)
    raise ValueError(scheme)


def ridge(Xtr, Ytr, lam):
    Xtr = Xtr.astype(np.float64); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Z = (Xtr - mu) / sd
    Wr = np.linalg.solve(Z.T @ Z + lam * len(Z) * np.eye(Z.shape[1]), Z.T @ (Ytr - Ytr.mean(0)))
    return lambda X: ((X - mu) / sd) @ Wr + Ytr.mean(0)


def nn_decode(pred, wte_n):
    pn = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-12)
    return (pn @ wte_n.T).argmax(1)


def run(ntrain=96, ntest=24, seed=0):
    tok, g = Tokenizer(), GPT2()
    wte = g.wte.astype(np.float64)
    wte_n = wte / np.linalg.norm(wte, axis=1, keepdims=True)
    trp = prompts(tok, os.path.join(A, "train.txt"), ntrain, seed)
    tep = prompts(tok, os.path.join(A, "test.txt"), ntest, seed + 1)
    t0 = time.time()
    cache = os.path.join(OUT, f"acts_{ntrain}_{ntest}.npz")
    if os.path.exists(cache):
        z = np.load(cache); tra = {op: list(z["tr_" + op]) for op in OPS}; tea = {op: list(z["te_" + op]) for op in OPS}
    else:
        tra, tea = capture(g, trp), capture(g, tep)
        np.savez(cache, **{"tr_" + op: np.stack(tra[op]).astype(np.float32) for op in OPS}, **{"te_" + op: np.stack(tea[op]).astype(np.float32) for op in OPS})
        tra = {op: [x.astype(np.float64) for x in tra[op]] for op in OPS}; tea = {op: [x.astype(np.float64) for x in tea[op]] for op in OPS}
    print("captured", time.time() - t0, flush=True)
    E = np.load("/home/claude/code/realcode_8_4_2.npy")
    # worst coalition (largest leakage gain)
    gains = {S: np.linalg.norm(np.linalg.solve(E[list(S), K:], E[list(S), :K]), 2) for S in itertools.combinations(range(N), T)}
    worstC = list(max(gains, key=gains.get))
    b = M // K
    trtok = np.stack(trp).reshape(-1, K, b)       # [P, K, b]
    tetok = np.stack(tep).reshape(-1, K, b)
    # prior baseline: always guess the most frequent training token
    vals, cnts = np.unique(trtok, return_counts=True)
    prior = float((tetok == vals[cnts.argmax()]).mean())
    results = {"prior_most_frequent": prior, "N": N, "k": K, "T": T, "m": M, "coalition": worstC,
               "ntrain_tokens": int(trtok.size), "ntest_tokens": int(tetok.size), "ops": {}}
    schemes = [("uncoded", None)] + [("analog", bt) for bt in BETAS] + [("ff", None)]
    for op in OPS:
        results["ops"][op] = {}
        for sch, bt in schemes:
            rng = np.random.default_rng(123)
            def build(acts, toks):
                Xs, Ys = [[] for _ in range(K)], [[] for _ in range(K)]
                for X, tk in zip(acts, toks):
                    v = views(X, sch, rng, E=E, coalition=worstC, beta=bt)
                    for j in range(K):
                        Xs[j].append(v[j] if sch == "uncoded" else v)
                        Ys[j].append(tk[j])
                return [np.concatenate(x) for x in Xs], [np.concatenate(y) for y in Ys]
            Xtr, Ytr = build(tra[op], trtok)
            Xte, Yte = build(tea[op], tetok)
            accs = []
            for j in range(K):
                best = 0.0
                # lambda chosen on a held-out slice of the TRAINING data only
                ntr = len(Xtr[j]); cut = int(0.85 * ntr)
                lam_best, acc_best = None, -1
                for lam in [1e-4, 1e-3, 1e-2, 1e-1]:
                    f = ridge(Xtr[j][:cut], wte[Ytr[j][:cut]], lam)
                    a = float((nn_decode(f(Xtr[j][cut:]), wte_n) == Ytr[j][cut:]).mean())
                    if a > acc_best: lam_best, acc_best = lam, a
                f = ridge(Xtr[j], wte[Ytr[j]], lam_best)
                accs.append(float((nn_decode(f(Xte[j]), wte_n) == Yte[j]).mean()))
            key = sch if bt is None else f"analog_beta{bt:g}"
            leak = None
            if sch == "analog" and bt > 0:
                leak = gaussian_leakage_per_entry(E, K, T, bt)
            results["ops"][op][key] = {"acc": float(np.mean(accs)), "acc_blocks": accs, "leak_per_entry_nats": leak}
            print(op, key, results["ops"][op][key], f"{time.time()-t0:.0f}s", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(results, open(os.path.join(OUT, "attack_privacy.json"), "w"), indent=1)


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 96, int(sys.argv[2]) if len(sys.argv) > 2 else 24)
