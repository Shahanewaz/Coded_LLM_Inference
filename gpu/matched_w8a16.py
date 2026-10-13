"""Matched-subset check for the W8A16 claim (reviewer request).

Re-evaluates BF16, W8A8 and W8A16 on exactly the same examples: the first N
examples (default 200) of every task, i.e. the subset W8A16 used in the paper.
The task lists are the fixed random subsets of cpd_gpu2.get_tasks (seed 0), so
the first N examples are identical across the three configurations.

Sanity check: the W8A16 numbers printed here must equal the W8A16 numbers
already in the paper (Table 13). If they do not, do NOT use the output.

Usage (RTX 5070, about 5-10 minutes per small model):
    python matched_w8a16.py --models HuggingFaceTB/SmolLM2-1.7B
    python matched_w8a16.py --models Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B HuggingFaceTB/SmolLM2-1.7B
Output: results/results_matched_w8a16.json
"""
import argparse, json, os, time
import torch
import cpd_gpu2 as G

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["HuggingFaceTB/SmolLM2-1.7B"])
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--task-n", type=int, default=500, help="must match the paper's task subset size (500)")
args = ap.parse_args()

tasks = G.get_tasks(args.task_n)
out = {}
os.makedirs("results", exist_ok=True)
for name in args.models:
    tok, model, state = G.load_model(name)
    r = out.setdefault(name, {})
    for tag, mode, ab in [("bf16", "plain", 8), ("w8a8", "wq", 8), ("w8a16", "wq", 16)]:
        G.set_mode(state, mode, abits=ab); t = time.time()
        res = G.eval_tasks(model, tok, tasks, limit=args.n)
        for v in res.values():
            v.pop("logprobs", None)
        # paper metric per task: acc for arc_easy, boolq, winogrande, mmlu; acc_norm for the others
        metric = {tn: (v["acc_norm"] if tn in ("arc_challenge", "hellaswag", "piqa") else v["acc"]) for tn, v in res.items()}
        r[tag] = {"per_task": res, "paper_metric": metric, "avg": sum(metric.values()) / len(metric)}
        print(name, tag, {k: round(100 * v, 1) for k, v in metric.items()}, "avg", round(100 * r[tag]["avg"], 1),
              f"{time.time() - t:.0f}s", flush=True)
    del model; torch.cuda.empty_cache()
    json.dump(out, open("results/results_matched_w8a16.json", "w"), indent=1)
print("done -> results/results_matched_w8a16.json")
