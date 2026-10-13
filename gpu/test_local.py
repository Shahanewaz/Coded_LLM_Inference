"""Offline test of cpd_gpu.py with a tiny random Qwen2 model (no downloads)."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpd_gpu as C
from transformers import Qwen2Config, Qwen2ForCausalLM

torch.manual_seed(0)
cfg = Qwen2Config(vocab_size=500, hidden_size=128, intermediate_size=344, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
model = Qwen2ForCausalLM(cfg).to(C.DEV).eval()
state = {"mode": "plain"}
n = C.wrap_model(model, state)
print("wrapped", n)
ids = torch.randint(0, 500, (1, 70))
outs = {}
def go(tag, **st):
    state.clear(); state.update(st)
    with torch.no_grad():
        outs[tag] = model(ids).logits.float()
go("plain", mode="plain")
go("w8a8", mode="w8a8", abits=8)
c = C.RNSCode(8, 4, 2, abits=8, seed=1); go("rns", mode="rns", code=c, abits=8)
print("rns stats", c.stats, "equal to w8a8:", torch.equal(outs["rns"], outs["w8a8"]))
c = C.RNSCode(8, 4, 2, abits=8, seed=5); c.byzantine = 3; go("byz", mode="rns", code=c)
print("byz stats", c.stats, "equal:", torch.equal(outs["byz"], outs["w8a8"]))
c = C.RNSCode(8, 4, 2, abits=8, seed=5); c.byzantine = 3; c.verify = False; c.check_exact = False; go("naive", mode="rns", code=c)
print("naive differs:", not torch.equal(outs["naive"], outs["w8a8"]))
go("w8a16", mode="w8a8", abits=16)
c = C.RNSCode(8, 4, 2, abits=16, seed=2); go("rns16", mode="rns", code=c, abits=16)
print("rns16 stats", c.stats, "equal:", torch.equal(outs["rns16"], outs["w8a16"]))
E = C.make_real_code(iters=30, restarts=1)
for fmt, beta in [("bf16", 10.0), ("fp32", 1e4), ("fp16", 1e4)]:
    a = C.AnalogCode(E, 4, 2, fmt, beta); go("an", mode="analog", code=a)
    print(fmt, beta, "max|diff| vs plain", (outs["an"] - outs["plain"]).abs().max().item(), "overflow", a.overflow,
          "leak", C.worst_leak_per_entry(E, 4, 2, beta))
print("w8a8 vs plain max diff", (outs["w8a8"] - outs["plain"]).abs().max().item())
w = [torch.randint(0, 500, (64,)) for _ in range(2)]
state.clear(); state.update(mode="plain"); print("ppl", C.ppl(model, w))
