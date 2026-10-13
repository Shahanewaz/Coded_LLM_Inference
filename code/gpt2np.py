"""Minimal NumPy GPT-2 (124M) with a pluggable executor for every public-weight linear layer.

Every matrix product of the form  Y = X @ W + b  where W is a *public* weight and X is a
*private* activation is routed through `linear(name, X, W, b)`.  Attention scores/values,
LayerNorm, GELU and softmax stay with the trusted coordinator.
"""
import json, os
import numpy as np
import regex as re

ASSETS = os.environ.get("CPD_ASSETS", "/home/claude/assets")


# ---------------------------------------------------------------- tokenizer (GPT-2 byte-level BPE)
def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))


class Tokenizer:
    def __init__(self, assets=ASSETS):
        self.encoder = json.load(open(os.path.join(assets, "encoder.json")))
        self.decoder = {v: k for k, v in self.encoder.items()}
        merges = open(os.path.join(assets, "vocab.bpe"), encoding="utf-8").read().split("\n")[1:-1]
        self.bpe_ranks = {tuple(m.split()): i for i, m in enumerate(merges)}
        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.cache = {}

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
        if not pairs:
            return [token]
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new, i = [], 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new.extend(word[i:]); break
                new.extend(word[i:j]); i = j
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new.append(first + second); i += 2
                else:
                    new.append(word[i]); i += 1
            word = tuple(new)
            if len(word) == 1:
                break
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
        self.cache[token] = list(word)
        return list(word)

    def encode(self, text):
        ids = []
        for tok in re.findall(self.pat, text):
            tok = "".join(self.byte_encoder[b] for b in tok.encode("utf-8"))
            ids.extend(self.encoder[t] for t in self.bpe(tok))
        return ids

    def decode(self, ids):
        text = "".join(self.decoder[i] for i in ids)
        return bytearray(self.byte_decoder[c] for c in text).decode("utf-8", errors="replace")


# ---------------------------------------------------------------- model
def _ln(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def plain_linear(name, X, W, b):
    Y = X @ W
    return Y + b if b is not None else Y


class GPT2:
    n_layer, n_head, d = 12, 12, 768

    def __init__(self, path=os.path.join(ASSETS, "gpt2_weights.npz")):
        z = np.load(path)
        self.w = {k: z[k].astype(np.float32) for k in z.files}
        self.wte = self.w["transformer.wte.weight"]
        self.wpe = self.w["transformer.wpe.weight"]

    def P(self, l, name):
        return self.w[f"transformer.h.{l}.{name}"]

    def linear_layers(self):
        """(name, W, b) for every outsourced public-weight GEMM, in execution order."""
        out = []
        for l in range(self.n_layer):
            out += [(f"h{l}.attn.c_attn", self.P(l, "attn.c_attn.weight"), self.P(l, "attn.c_attn.bias")),
                    (f"h{l}.attn.c_proj", self.P(l, "attn.c_proj.weight"), self.P(l, "attn.c_proj.bias")),
                    (f"h{l}.mlp.c_fc", self.P(l, "mlp.c_fc.weight"), self.P(l, "mlp.c_fc.bias")),
                    (f"h{l}.mlp.c_proj", self.P(l, "mlp.c_proj.weight"), self.P(l, "mlp.c_proj.bias"))]
        return out

    def forward(self, ids, linear=plain_linear, hook=None, upto_layer=None, return_hidden=False):
        """ids: 1-D int array (one sequence). Returns logits [m, V] (float32).
        hook(name, X) is called with every linear-layer input (for attacks / statistics).
        If upto_layer is given, stop after that many blocks and return the residual stream."""
        m = len(ids)
        h = self.wte[ids] + self.wpe[np.arange(m)]
        H, dh = self.n_head, self.d // self.n_head
        mask = np.triu(np.full((m, m), -1e10, dtype=np.float32), 1)
        L = self.n_layer if upto_layer is None else upto_layer
        for l in range(L):
            x = _ln(h, self.P(l, "ln_1.weight"), self.P(l, "ln_1.bias"))
            nm = f"h{l}.attn.c_attn"
            if hook: hook(nm, x)
            qkv = linear(nm, x, self.P(l, "attn.c_attn.weight"), self.P(l, "attn.c_attn.bias")).astype(np.float32)
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(m, H, dh).transpose(1, 0, 2)
            k = k.reshape(m, H, dh).transpose(1, 0, 2)
            v = v.reshape(m, H, dh).transpose(1, 0, 2)
            att = q @ k.transpose(0, 2, 1) / np.sqrt(dh) + mask          # private x private: stays local
            att = np.exp(att - att.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            a = (att @ v).transpose(1, 0, 2).reshape(m, self.d)
            nm = f"h{l}.attn.c_proj"
            if hook: hook(nm, a)
            h = h + linear(nm, a, self.P(l, "attn.c_proj.weight"), self.P(l, "attn.c_proj.bias")).astype(np.float32)
            x = _ln(h, self.P(l, "ln_2.weight"), self.P(l, "ln_2.bias"))
            nm = f"h{l}.mlp.c_fc"
            if hook: hook(nm, x)
            f = _gelu(linear(nm, x, self.P(l, "mlp.c_fc.weight"), self.P(l, "mlp.c_fc.bias")).astype(np.float32))
            nm = f"h{l}.mlp.c_proj"
            if hook: hook(nm, f)
            h = h + linear(nm, f, self.P(l, "mlp.c_proj.weight"), self.P(l, "mlp.c_proj.bias")).astype(np.float32)
        if upto_layer is not None or return_hidden:
            return h
        h = _ln(h, self.w["transformer.ln_f.weight"], self.w["transformer.ln_f.bias"])
        return h @ self.wte.T            # tied LM head (computed locally; see paper)


def log_softmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))
