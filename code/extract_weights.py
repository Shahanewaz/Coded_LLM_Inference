"""Convert the ONNX-zoo GPT-2 (gpt2-lm-head-10.onnx) into gpt2_weights.npz."""
import os, sys, numpy as np, onnx
from onnx import numpy_helper
A = os.environ.get("CPD_ASSETS", "/home/claude/assets")
m = onnx.load(os.path.join(A, "gpt2-lm-head-10.onnx"))
w = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
np.savez(os.path.join(A, "gpt2_weights.npz"), **{k: v for k, v in w.items() if k.startswith("transformer.") and not k.endswith(".attn.bias")})
