"""Print every number used in the paper's tables from results/*.json."""
import json, glob, os
R = "/home/claude/code/results"
J = lambda f: json.load(open(os.path.join(R, f)))

print("== Quality (WikiText-2 PPL; LAMBADA acc) ==")
for f in sorted(glob.glob(os.path.join(R, "ppl_*.json"))):
    d = json.load(open(f))
    print(f"{os.path.basename(f):40s} ppl={d.get('ppl', float('nan')):10.4f} tokens={d.get('tokens')} "
          f"lambada={d.get('lambada_acc')} exact={d.get('ff_all_exact')} ops={d.get('ff_ops')} nonfinite={d.get('nonfinite_windows')}")

print("\n== Privacy attack: top-1 token recovery (%) ==")
d = J("attack_privacy.json")
print("prior (most frequent token):", round(100 * d["prior_most_frequent"], 1), "| coalition", d["coalition"])
for op, v in d["ops"].items():
    print(op, {k: round(100 * x["acc"], 1) for k, x in v.items()})

print("\n== Byzantine (RNS) ==")
print(json.dumps(J("attack_integrity_ff.json")["ff"], indent=1))
print("\n== Byzantine (analog residual test) ==")
for k, v in J("attack_integrity_analog.json").items():
    print(k, {m: v[m] for m in ("clean", "steer", "stealth")})

print("\n== Barrier ==")
b = J("barrier.json")
print("kappa", b["kappa"], "w8a8 rel mse", b["w8a8_rel_mse"], "ff exact", b["ff_exact_vs_w8a8"])
for r in b["rows"]:
    print({k: (f"{v:.3g}" if isinstance(v, float) else v) for k, v in r.items()})

print("\n== Cost ==")
print(json.dumps(J("cost.json"), indent=1))
