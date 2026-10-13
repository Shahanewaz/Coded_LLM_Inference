# Code for "Quantize the Secret, Not the Share"

Everything runs on CPU with NumPy. Worker arithmetic is emulated bit-exactly: INT8×INT8→INT32 GEMMs are computed as exact integer sums in float64, and BF16/FP16 GEMMs round their inputs, accumulate in FP32 and round their outputs.

## Assets (`CPD_ASSETS`, default `/home/claude/assets`)
- `gpt2-lm-head-10.onnx` from the ONNX model zoo (GitHub LFS: `onnx/models/validated/text/machine_comprehension/gpt-2/model/`). It is converted to `gpt2_weights.npz`; see the snippet in `gpt2np.py`'s docstring or `extract_weights.py`.
- GPT-2 BPE files `encoder.json` and `vocab.bpe`.
- WikiText-2, word-level `train.txt` and `test.txt` (pytorch/examples).
- `lambada_test.jsonl` (OpenAI LAMBADA test set).

## Files
| file | purpose |
|---|---|
| `gpt2np.py` | GPT-2 forward pass with a pluggable executor for every public-weight GEMM |
| `coded.py` | `FFCode` (RNS secret-shared polynomial code, exact, RS check), `AnalogCode` (Gaussian-masked real code), quantizers, format emulation |
| `realcode.py` | search for the analog baseline code (writes `realcode_8_4_2.npy`) |
| `barrier.py` | Theorem 1 check on a real GEMM → `results/barrier.json` |
| `evalppl.py` | WikiText-2 perplexity and LAMBADA accuracy for any executor (`fp32`, `bf16`, `w8a8`, `ff:N,k,T,abits`, `analog:fmt,beta`) |
| `attack_privacy.py` | learned token-inversion attack on coalition views → `results/attack_privacy.json` |
| `attack_integrity.py` | Byzantine steering attack (naive vs Reed–Solomon; analog residual test) → `results/attack_integrity.json` |
| `cost.py` | offloadable-FLOP fractions, residue counts, throughput model → `results/cost.json` |
| `wsplit_test.py` | exactness check for the weight-split code (App. B) |
| `tightness.py` | Thm. 1(b): the tight code ($N$=9, $T$=2) on Gaussian data or GPT-2 activations (`python tightness.py gaussian|gpt2`) → `results/tightness_*.json` |
| `gamma_corr.py` | correlation factor $\gamma_w$ of Lemma 2 on GPT-2 activations → `results/gamma_gpt2.json` |
| `frontier.py` | Fig. 3 (privacy vs quality on Qwen2.5-1.5B, from the GPU results) |
| `plots.py` | figures |
| `run_quality2.sh` | the quality runs reported in the paper |
| `make_tables.py` | prints the numbers that go into the paper's tables |

Seeds are fixed inside each script.
