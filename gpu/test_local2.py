"""Offline end-to-end test of cpd_gpu2.py: tiny random models, fake tokenizer and fake datasets."""
import sys, os, types, random, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# simulate CUDA: no int64 matmul
_td, _mm = torch.tensordot, torch.Tensor.__matmul__
def td(a, b, *x, **k):
    if a.dtype == torch.int64 or b.dtype == torch.int64: raise NotImplementedError("int64 matmul")
    return _td(a, b, *x, **k)
def mm(a, b):
    if a.dtype == torch.int64 or b.dtype == torch.int64: raise NotImplementedError("int64 matmul")
    return _mm(a, b)
torch.tensordot = td; torch.Tensor.__matmul__ = mm
import cpd_gpu2 as C
from transformers import Qwen2Config, Qwen2ForCausalLM, LlamaConfig, LlamaForCausalLM


class Enc:
    def __init__(self, ids): self.input_ids = ids


class FakeTok:
    eos_token_id = 0
    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = [3 + (ord(c) % 250) for c in text]
        return Enc(torch.tensor([ids]) if return_tensors == "pt" else ids)
    def decode(self, ids):
        return "".join(chr(((i - 3) % 250) + 32) if i >= 3 else "" for i in ids)


def fake_load_model(name):
    torch.manual_seed(0)
    if "Smol" in name:
        cfg = LlamaConfig(vocab_size=260, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048, attn_implementation="eager")
        model = LlamaForCausalLM(cfg)
    else:
        cfg = Qwen2Config(vocab_size=260, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048, attn_implementation="eager")
        model = Qwen2ForCausalLM(cfg)
    model = model.to(torch.bfloat16).to(C.DEV).eval()
    state = {"mode": "plain", "attn": "plain"}
    C.wrap_model(model, state)
    return FakeTok(), model, state


class FakeDS(list):
    def __getitem__(self, i):
        if isinstance(i, str):
            return [r[i] for r in self]
        return list.__getitem__(self, i)


def fake_ds(name, *a, split=None, **k):
    R = random.Random(hash((name, split)) % 1000)
    words = "the cat sat on a mat while dogs ran far away and birds sang loudly in trees".split()
    sent = lambda n: " ".join(R.choice(words) for _ in range(n))
    if "wikitext" in name:
        return FakeDS([{"text": sent(400)} for _ in range(30)])
    if "arc" in name:
        return FakeDS([{"question": sent(8) + "?", "choices": {"text": [sent(3) for _ in range(4)], "label": list("ABCD")}, "answerKey": "B"} for _ in range(20)])
    if "hellaswag" in name:
        return FakeDS([{"activity_label": "Cooking", "ctx_a": sent(6), "ctx_b": sent(3), "endings": [sent(4) for _ in range(4)], "label": "2"} for _ in range(20)])
    if "boolq" in name:
        return FakeDS([{"passage": sent(30), "question": sent(6), "answer": bool(R.random() > .5)} for _ in range(20)])
    if "winogrande" in name:
        return FakeDS([{"sentence": sent(5) + " _ " + sent(4), "option1": "Tom", "option2": "Ann", "answer": "1"} for _ in range(20)])
    if "piqa" in name:
        raise FileNotFoundError("script dataset")
    if "mmlu" in name:
        return FakeDS([{"question": sent(10), "choices": [sent(2) for _ in range(4)], "answer": 1} for _ in range(20)])
    if "gsm8k" in name:
        return FakeDS([{"question": sent(12), "answer": sent(10) + " #### 42"} for _ in range(10)])
    raise FileNotFoundError(name)


C.load_model = fake_load_model
C.ds_load = fake_ds
os.chdir(os.path.dirname(os.path.abspath(__file__)))
for f in os.listdir("."):
    if f.startswith("quick_results2") or f == "cpd_run2.log":
        os.remove(f)
sys.argv = ["x", "--quick", "--stages", "stress", "tasks", "attention", "generate", "privacy"]
C.main()
for st in ["stress", "tasks", "attention", "generate", "privacy"]:
    f = f"quick_results2_{st}.json"
    if os.path.exists(f):
        d = json.load(open(f))["result"]
        print("==", st, json.dumps(d if st != "stress" else d["summary"], default=str)[:1500])
    else:
        print("== MISSING", st, [x for x in os.listdir(".") if "FAILED" in x])
