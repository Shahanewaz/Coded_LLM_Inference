# Coded LLM Inference

**Privacy-preserving, verifiable LLM inference using quantization, residue number systems (RNS), and polynomial codes.**

This repository contains CPU reference experiments and GPU implementations for **RNS-coded inference**. The implementation quantizes model operands *before* secret sharing, distributes encoded matrix multiplications across logical workers, and reconstructs their outputs using finite-field arithmetic. It also evaluates analog masking baselines, output fidelity, privacy attacks, adversarial worker behavior, and computational cost.

> **Execution status:** GPU workers are currently **emulated sequentially on a single GPU**, not deployed as separate networked machines. The code demonstrates arithmetic correctness and evaluates components of the protocol; it does not establish an end-to-end distributed-inference speedup.

## How it works

For a quantized matrix multiplication, the coded execution path follows this pattern:

1. **Quantize:** Convert activation and weight operands to integer representations.
2. **Encode:** Split operands into blocks, add secret-sharing randomness, and evaluate polynomial shares over suitable finite fields.
3. **Compute:** Have logical workers multiply encoded shares using INT8 × INT8 → INT32 matrix multiplication.
4. **Recover:** Decode the returned products using polynomial interpolation and residue reconstruction.
5. **Verify:** Use codeword-consistency / Reed–Solomon checks to identify or correct supported worker errors, or abort when consistency cannot be established.
6. **Continue inference:** Feed the recovered result into the model's remaining operations.

The integer-coded path targets **bit-exact agreement with the chosen quantized reference computation**, not necessarily the unquantized BF16/FP32 model. Privacy and integrity statements depend on the protocol's stated threat model and coding parameters.

## Repository structure

```text
Coded_LLM_Inference/
├── README.md
├── code/                      # CPU reference and analysis (NumPy)
│   ├── README.md              # CPU-specific assets and experiment details
│   ├── coded.py               # Finite-field coding, analog coding, quantization
│   ├── gpt2np.py              # GPT-2 forward pass with pluggable GEMM executor
│   ├── evalppl.py             # Perplexity and LAMBADA evaluation
│   ├── attack_privacy.py      # Token-inversion experiments
│   ├── attack_integrity.py    # Adversarial-worker experiments
│   ├── barrier.py             # Analog precision/privacy experiments
│   ├── tightness.py           # Analog-code tightness experiments
│   ├── gamma_corr.py          # Activation correlation measurements
│   ├── kappa_check.py         # Rounding-error diagnostics
│   ├── cost.py                # Compute / communication cost model
│   ├── make_tables.py         # Tabulates experiment results
│   └── results/               # Saved CPU outputs
└── gpu/                       # PyTorch / CUDA experiments
    ├── cpd_gpu.py             # Single-GPU coded inference and baselines
    ├── cpd_gpu2.py            # Additional coded execution / experiment stages
    ├── cpd_ada.py             # Larger-model experiments on 32 GB GPU
    ├── cpd_diag.py            # GPU diagnostics
    ├── matched_w8a16.py      # Matched-setup BF16 / W8A8 / W8A16 comparison
    ├── run_main.bat           # Windows experiment sequence
    ├── run_diag.bat           # Windows diagnostics sequence
    ├── results/               # GPU experiment outputs
    └── results_ada/           # RTX 5000 Ada experiment outputs
```

The tree highlights the primary entry points; additional scripts and artifacts are included in each directory.

## Getting started

### Prerequisites

- Python 3 and a virtual environment
- **CPU experiments:** NumPy and the GPT-2 / benchmark assets described in [`code/README.md`](code/README.md)
- **GPU experiments:** a supported NVIDIA GPU, CUDA-compatible PyTorch, and the Hugging Face libraries required by the experiment scripts
- Enough GPU memory for the selected model and execution mode; the larger-model scripts were developed for an **RTX 5000 Ada (32 GB)**, while `cpd_gpu.py` targets an **RTX 5070 (12 GB)**

> This repository does not currently provide a pinned, top-level dependency manifest. Install versions compatible with your hardware and consult import statements if an optional experiment requires additional packages. Some datasets and pretrained model assets are downloaded separately.

### Clone the repository

```bash
git clone https://github.com/Shahanewaz/Coded_LLM_Inference.git
cd Coded_LLM_Inference
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Or on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

For GPU experiments, install a CUDA-compatible PyTorch build following the [official PyTorch installation guide](https://pytorch.org/get-started/locally/), together with the Python packages imported by the selected script.

## Run the GPU experiments

Commands below are run from the `gpu/` directory:

```bash
cd gpu
```

**Quick test (RTX 5070-oriented path):**

```bash
python cpd_gpu.py --quick
```

**Main single-GPU experiment:**

```bash
python cpd_gpu.py
```

This path checks quantized and coded model quality, integer GEMM reconstruction, malicious-worker handling, and kernel timing. The script writes `results_gpu.json` in its working directory.

**Quick larger-model test (RTX 5000 Ada-oriented path):**

```bash
python cpd_ada.py --quick
```

**Selected larger-model evaluation:**

```bash
python cpd_ada.py --tag q7 --stages quality --models Qwen/Qwen2.5-7B --analog
```

**Run coded attention and quality checks:**

```bash
python cpd_ada.py --tag q7 --stages quality attention --models Qwen/Qwen2.5-7B --analog
```

**Measure GPU timing:**

```bash
python cpd_ada.py --tag hw --stages timing
```

On Windows, `gpu/run_main.bat` contains the multi-stage experiment sequence used for the larger-model runs. It includes quality, attention, downstream tasks, privacy, and generation checks. Experiment artifacts are saved as stage-specific JSON files (for example, `results3_<tag>_<stage>.json`); previously completed stages may be skipped when rerunning the same tag.

These runs may download pretrained models and evaluation datasets. Available stages and options vary between `cpd_gpu.py`, `cpd_gpu2.py`, and `cpd_ada.py`.

## Run the CPU experiments

The CPU implementation is useful for inspecting the coding arithmetic without CUDA. It uses NumPy to emulate integer GEMMs exactly and to model lower-precision floating-point baselines.

Follow the asset setup in [`code/README.md`](code/README.md). In particular, the GPT-2 experiments expect a converted GPT-2 weight archive, tokenizer assets, and benchmark files. The CPU scripts reference `CPD_ASSETS` (whose documented default is `/home/claude/assets`), so **set that environment variable for your machine**.

From `code/`, representative experiment entry points are:

```bash
cd code
python barrier.py
python cost.py
python attack_privacy.py
python attack_integrity.py
python tightness.py gaussian
```

For model evaluation, `evalppl.py` supports executor identifiers such as `fp32`, `bf16`, `w8a8`, `ff:N,k,T,abits`, and `analog:fmt,beta`. Refer to its argument parser and [`code/README.md`](code/README.md) before selecting a configuration. Dataset-dependent scripts will not run until their assets are configured.

## Experiment coverage

| Area | Implementation / entry point | What is checked |
| --- | --- | --- |
| Exact coded products | `code/coded.py`, `gpu/cpd_gpu.py`, `gpu/cpd_gpu2.py` | Agreement with the integer quantized reference |
| Language-model fidelity | `code/evalppl.py`, `gpu/cpd_ada.py` | Perplexity, task scores, and generation equivalence |
| Privacy evaluation | `code/attack_privacy.py`, GPU privacy stage | Token-recovery attacks on worker / coalition observations |
| Byzantine integrity | `code/attack_integrity.py`, GPU quality stage | Detection and correction of supported tampering patterns |
| Analog-code comparisons | `code/barrier.py`, `code/tightness.py`, GPU analog mode | Precision and privacy behavior of floating-point shares |
| Hardware and cost | `code/cost.py`, GPU timing stage | Worker GEMM performance and cost components |
| Diagnostics | `gpu/cpd_diag.py`, `gpu/matched_w8a16.py` | Integer throughput and matched quantization comparisons |

Reported experiments span GPT-2 CPU emulation and GPU tests on Hugging Face LLMs, including Qwen2.5 and OLMo-2 variants. Saved outputs are located under [`code/results/`](code/results/), [`gpu/results/`](gpu/results/), and [`gpu/results_ada/`](gpu/results_ada/).

## Reproducibility notes

- **Worker simulation:** multiple logical workers execute on one device in the current GPU setup; communication over a real network has not been measured.
- **Exactness reference:** reconstruction is compared with the same quantized execution, not with ideal real-valued inference.
- **Quantization comparisons:** BF16, W8A8, and W8A16 are separate execution modes; use `gpu/matched_w8a16.py` for matched-example comparisons.
- **Integrity:** error correction has a finite detection / correction radius and is not a guarantee against arbitrary unbounded tampering.
- **Privacy:** conclusions apply to the specified worker-protocol view and assumptions. Model outputs, deployment side channels, and external timing leakage are outside the demonstrated protection.
- **Performance:** encoding, decoding, and communication matter. Current results should not be interpreted as production or distributed-system speedups.
- **Configuration:** inspect individual script flags, local paths, random seeds, and cached-result behavior before reproducing a particular result.

## Current limitations and next steps

The present implementation emphasizes **correctness, privacy experiments, and integrity checks** rather than deployment readiness. Natural extensions include real multi-worker execution, measured network overhead, optimized/fused coordinator encoding and decoding, and a pinned environment with automated reproducibility tests.

## Reference

Research project: **“Quantize the Secret, Not the Share.”**

Repository: https://github.com/Shahanewaz/Coded_LLM_Inference

For code-specific details, start with [`code/README.md`](code/README.md) and the docstrings of the corresponding GPU entry points.
