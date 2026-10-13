"""Search for a good real-valued (analog) code: minimize amp_worst * sqrt(gain_worst), the factor
that multiplies the floating-point error at a fixed per-entry Gaussian leakage (see Theorem 1)."""
import itertools, numpy as np

def metrics(E, k, T):
    N = E.shape[0]
    amp = max(np.linalg.norm(np.linalg.pinv(E[list(S)])[:k], 2) for S in itertools.combinations(range(N), k + T))
    gain = max(np.linalg.norm(np.linalg.solve(E[list(S), k:], E[list(S), :k]), 2) ** 2 for S in itertools.combinations(range(N), T))
    return amp, gain

def search(N, k, T, seed=0, iters=1500, restarts=4):
    rng = np.random.default_rng(seed)
    best, bv = None, np.inf
    for r in range(restarts):
        E = rng.standard_normal((N, k + T)); E /= np.linalg.norm(E, axis=1, keepdims=True)
        a, g = metrics(E, k, T); v = a * np.sqrt(g); step = 0.3
        for it in range(iters):
            F = E + step * rng.standard_normal(E.shape); F /= np.linalg.norm(F, axis=1, keepdims=True)
            a2, g2 = metrics(F, k, T); v2 = a2 * np.sqrt(g2)
            if v2 < v:
                E, v = F, v2
            elif it % 200 == 199:
                step *= 0.7
        if v < bv:
            best, bv = E, v
    return best

if __name__ == "__main__":
    E = search(8, 4, 2)
    print(metrics(E, 4, 2))
    np.save("/home/claude/code/realcode_8_4_2.npy", E)
