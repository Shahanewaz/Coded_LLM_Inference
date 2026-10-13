# Quantize the Secret, Not the Share: NAACL/ARR draft, version 3.8

**Code repository (placeholder):** https://github.com/ANONYMIZED/rns-coded-llm
Replace this URL in two places in the paper: the last sentence of the abstract (`paper/parts2/intro.tex`) and App. K, Reproducibility (`paper/parts2/appendix_tables.tex`). Both are marked `% TODO`. For the anonymous review version, use an anonymized mirror (e.g. anonymous.4open.science).

## Folders
- **`paper/`**: LaTeX sources (official ACL style, `[review]` mode) and the compiled `main.pdf`.
  - The main text ends on page 8. Limitations, Ethics, references and the appendix (proofs plus supplementary results) follow.
  - `main.tex` is assembled from `parts2/` by `build3.py`. Shared pieces (macros, benchmark rows, the weight-split appendix) still come from `parts/`.
  - You can also edit `main.tex` directly.
  - Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main`.
- **`code/`**: CPU experiments (GPT-2, bit-exact NumPy emulation). See `code/README.md`.
- **`gpu/`**: GPU experiments, run on an RTX 5070 under Windows. Raw results and logs are in `gpu/results/`.

## What changed in version 3.8 (response to the reviewer-style comments)
- **Novelty positioning.**
  - The Contributions paragraph now opens by naming what is classical: secret sharing, polynomial codes, RS decoding, quantization and CRT, plus CodedPrivateML (quantize-then-share) and Ozaki II (CRT over INT8).
  - The contributions are reordered: (1) the tight barrier, (2) the complete transformer protocol with whole-session privacy that covers observable correct/abort decisions, (3) INT8-native RNS execution, presented as the integration it is, and (4) experiments.
- **Practical utility.**
  - A new "Scope and intended use" paragraph describes the target: a trusted, compute-poor coordinator (CPU or TEE) renting untrusted GPUs over a fast network.
  - A new network cost model, Eq. (5), gives rho_net = (C/B_net)(Nr/k)(d+h)/(2dh). Outsourcing pays only if C/B_net < ~800 FLOP/byte (Llama-3-8B FFN) or ~1,600 (70B).
  - New Table 8 breaks down one GEMM into measured worker and coordinator time and modeled bandwidth-bound and network time. With a GPU coordinator like ours, the network alone is 6–21x the local GEMM, which is stated plainly.
  - Limitations and Conclusion now say that no networked run was done and that it is the next step.
- **Abstract.** "on two GPUs" now reads "on two single-GPU testbeds with emulated workers". The PIN sentence also reports token recovery below the prior, framed as consistent with the theorem.
- **Barrier wording.** "Private analog inference needs 24 bits" is now conditional on Assumption 1, the certificates and the workloads. A new sentence says that a loose certificate is not evidence that an attacker can recover the prompt.
- **Fano and coalitions.** App. A.4 now says that a coalition needs a *joint* certificate, because per-worker bounds do not add up. The reported pair leakages are joint (log-det over the pair's combined shares).
- **W8A16.** The "W8A16 recovers SmolLM2's loss" claim was removed: the subsets do not match, and per-example log-probs were not saved. `gpu/matched_w8a16.py` reruns BF16, W8A8 and W8A16 on the identical first 200 examples per task. Its W8A16 output must reproduce Table 13's W8A16 rows as a sanity check.
- **PIN attack.** RQ2 now says that zero PIN recovery is a weak signal, since analog β=10 also recovers zero PINs at layer 0 while still recovering 27–28% of tokens. Privacy rests on Thm 3(b).

## What changed in version 3.7 (proofreading pass)
- **Compact theorem statements.**
  - Thm 3 now states (a) and (b) in four lines. Its assumptions (C1)–(C4) and the scope of the privacy claim (worker-protocol transcript, not the complete service) are in the new Remark 1, after the proof idea. App. A.8 cites (C1)–(C4) explicitly.
  - Thm 1(b) is one sentence; the definition of the factor in it moved to the proof idea.
  - Thm 2: the definition of the contraction dimension moved below the proof idea.
- **Notation.** The prompt is now written with a calligraphic P and the adversary's view with a calligraphic V, so they no longer clash with the attention matrices P and V.
- **Every float is referenced.** Fig. 1, Fig. 4, Prop. 4, and Tables 11, 12 and 17 had no in-text reference; they do now. App. F also had an empty heading, which now has a sentence.
- **Readable tables.**
  - Tables 14 (stress tests) and 16 (per-product error) are full-width at normal size, no longer shrunk.
  - Tables 2, 4 and 6, and the appendix timing tables (11, 12, 15), were restructured to fit the column at natural size.
  - Tables 7 and 10 are full-width.
  - Table 3 now has two clean panels: (a) exact coded inference and (b) analog codes, each with its own header. The numbers are unchanged.
- **Figures.**
  - All plots use a Times-metric serif font (Liberation Serif) with STIX math, inward ticks and a new palette (`code/pstyle.py`).
  - Fig. 3 was redrawn: there is a legend, β labels, a shaded "below prior" band, and labelled callouts for the two overlapping points. The data are unchanged.
  - The duplicated sentence in the Fig. 1 caption was removed, and the attention note inside Fig. 1 was clarified.
- **Corrections.**
  - The 8-bit attention perplexity change (−8.5% to +6.0%) is now stated against the right baseline: W8A8 projections with BF16 attention, with those numbers given in §5.
  - Table 15 now lists all 15 measured shapes. The quoted ranges (1.4–2.1×, 0.03–0.83×) come from rows that the old table had omitted.
  - "Attention is at most 13%" was changed to 13.3%.
- A GitHub placeholder was added; see the top of this file.

## What changed in version 3
1. **Three theorems instead of five, each with a basic part (a) and a stronger part (b).**
   - Thm 1, precision–privacy barrier:
     - (a) the lower bound;
     - (b) **new:** the bound is tight. An explicit code attains it up to a factor 1 + δ̄ → 1, which is verified numerically.
   - Thm 2, exact INT8 coded products:
     - (a) public weights;
     - (b) two private operands (attention).
   - Thm 3, private, exact and verifiable inference:
     - (a) token-for-token exactness, with correction or abort;
     - (b) whole-session privacy against mobile, adaptive coalitions.
   - Supporting results are in the appendix: Lemma 1 (polynomial codes), Prop. 1 (tight codes), Cor. 1 (whole-request barrier), Lemma 2 (correlated data; γ_w = 0.349 measured on GPT-2, which matches the observed 0.35), and Lemma 3 (k+T is optimal for any scheme with 1/k-size shares).
2. **ACL-style structure.**
   - Section order: Introduction → Setting and threat model → Analog barrier → Method → one Experiments section organized by research questions (RQ1 fidelity, RQ2 privacy, RQ3 integrity, RQ4 cost) → Related work → Conclusion.
   - Privacy and security are now woven through every section rather than kept in a separate section.
   - New Fig. 3 puts privacy and quality on one plot.
3. **Corrections from an internal review.**
   - The cost model now counts Nr/k bytes per element (ρ ≈ 0.16 on an RTX 5070, 0.47 on an H100).
   - The abort guarantee is stated against adversarial tampering only within the detection radius.
   - Smaller fixes: the contraction dimension in Thm 2, the W8A16 range, the residual-test wording, and several numbers.

## What changed in version 3.1
- **Theorem 1 is scoped.** The abstract, introduction and limitations state the model: linear codes, exact linear decoders, Gaussian-mask leakage certificates and probabilistic rounding. A new main-text "Scope" paragraph and App. A.5 show:
  - the error bound needs only second moments;
  - non-Gaussian masks can only increase the true leakage;
  - Assumption 1 holds with kappa >= 1/12 for continuous shares, and kappa is measured at 0.177-0.186 for BF16/FP16/FP32 shares of real GPT-2 activations (`code/kappa_check.py`);
  - the failure modes (exactly representable values, overflow) are listed.
- **Efficiency is framed as feasibility.** The abstract, introduction, RQ4 and conclusion say plainly that end-to-end speedups are not shown. Total worker work is 4x (projections) to 16x (attention) a local product, and the coordinator is unoptimized.
- **The NLP contribution comes first.** A new "Why language models are different" paragraph covers composition through nonlinearities, private x private attention, autoregressive adaptivity and per-token scales. The end-to-end transformer guarantee is now the first contribution.
- **Privacy is qualified.** Thm. 3(b) is labelled "perfect privacy within the threat model" and lists its assumptions. The threat model names what is outside the guarantee (deployment side channels, output text, weights).
- **Two recovery regimes.** A new paragraph and Table 2 separate one-sided codes (threshold k+T, 1/k work per worker) from two-sided codes (threshold 2(k+T)-1, full-size product per worker, 16x total). This affects the straggler/Byzantine budgets and cost.
- **Page budget.** The GPT-2 Byzantine table moved to App. F; the main text still ends on page 8.

## What changed in version 3.2
- **Rounding assumption.** The kappa >= 1/12 statement is now explicitly under the smooth-density approximation for round-to-nearest, with its failure cases.
- **Exactness vs. BF16 fidelity.** "Privacy never costs accuracy" is replaced by: secret sharing adds no reconstruction error beyond the chosen quantized reference model.
- **Theorem 3(b) assumptions are explicit.**
  - public schedule sigma (padded prompt length and number of decoding steps);
  - deviators lie in the current coalition;
  - the decoder decides only from codeword-consistency tests.
  - The proof shows why the correct/abort transcript is then input-independent. The workers' view excludes the output text and timing outside the protocol.
- **Name rendering.** "RNS-Coded" is now plain text instead of small caps; text extraction showed "RNS-C ODED".
- **GPT-2 malicious-worker table** is back in the main text. The main text is now about half a column over 8 pages; it will be compacted after the 7B results.
- **New App. B, "Beyond Linear Codes and Linear Layers":**
  - Prop. 2: nonlinear decoders can only reach B/(1+B), even when told the masks (any affine decoder under Assumption 1; any decoder under a CLT-justified Gaussian surrogate);
  - Prop. 3: no real-valued mask distribution gives perfect privacy, so a compact group, and hence quantization, is forced;
  - coding the nonlinear layers: Lagrange codes need polynomial surrogates, RNS lacks comparison/division, approximate codes lose exactness and privacy.

## What changed in version 3.3 (App. B corrected)
- **Prop. 2 (nonlinear decoders) states its assumptions.**
  - (a) any affine decoder, under Assumption 1;
  - (b) any decoder of one output entry, which needs Gaussian data plus a Gaussian surrogate for the accumulated rounding error;
  - (c) any decoder that sees the stored shares, which needs per-entry Gaussian errors, a stronger idealization.
  - Decoders that exploit deterministic rounding are explicitly not covered.
- **Prop. 3 is generalized and narrowed.** It holds for any mask matrix and any mask distribution, but only for additive masking. The step to quantization is labelled an argument, not a theorem.
- **Removed wrong claims:**
  - rotary embeddings "could be folded into a code";
  - Z_{2^32} "needs INT32 operands" (the correct reason is that only 2 evaluation points have invertible differences);
  - a citation misapplied to scaling/permutation blinding;
  - "any finite-precision torus element is fixed-point" (replaced by the finite-subgroup statement).

## What changed in version 3.4 (second external review)
- **A.** Prop. 2's Gaussian form is described as a modeling assumption motivated by error aggregation, not as a consequence of the central limit theorem. The "nonlinear decoders gain little" wording is removed.
- **B.** The main text now says "no nondegenerate additive real-valued masking scheme of the form in Prop. 3 ..."
- **C.** The adversary's view V is defined formally in Sec. 2 (shares, randomness, decoder outcomes, schedule-determined metadata). Thm. 3(b) refers to it, and the guarantee is labelled privacy of the worker-protocol transcript, not of the whole service.
- **D.** Integer-arithmetic implementation conditions are stated after Thm. 2. New App. A (implementation semantics) covers INT32 accumulation, the FP32-exact coordinator arithmetic, and exactly what our exactness checks verify.
- **Experiments.** The privacy result is phrased as "no attacker can beat the best prior-only strategy". GSM8K coded runs are described as verifying equivalence, not generation quality.
- **Final checks.** The analog-section title and numbers are conditional on Assumption 1.

## What changed in version 3.6 (final check)
- **Diagnostics are in.** New App. "Diagnostics on the RTX 5000 Ada":
  - INT8 GEMM throughput and the cost of the modular-reduction pass;
  - per-product analog error on real 1.5B/7B activations (never below the Theorem 1 bound);
  - controls for the below-BF16 perplexity effect (seeds, FP32, noise).
- **GSM8K appendix.** A 40-problem check with saved generations: extraction and truncation are not the cause, and the BF16/W8A8 gap did not reproduce (p=0.63). GSM8K is used only as an exactness check.
- **Full numerical audit against the raw JSON files.** All mismatches were fixed, including subset sizes, LAMBADA counts, the GPT-2 coded-window note, a timing rounding, the analog BF16 timing (not measured), and the Delta rounding note.
- **Claims tightened:**
  - the abort guarantee is limited to the detection radius;
  - the PIN claim is limited to the layers attacked and the input layer;
  - the Fig. 1 caption no longer says redundancy provides privacy;
  - the widest-FFN claim is removed;
  - the Limitations entry on quality at scale is qualified;
  - the operation index in Thm. 3 is renamed to tau.
- **New verified references:** THE-X (Findings ACL 2022), SecFormer (Findings ACL 2024), Split-and-Denoise (ICML 2024), QuaRot (NeurIPS 2024), BumbleBee (NDSS 2025), OLMo 2.

## What changed in version 3.5 (7B results, RTX 5000 Ada)
- **New models:** Qwen2.5-7B and OLMo-2-7B (a second, fully open family), run on one RTX 5000 Ada (32 GB). Raw results are in `gpu/results_ada/`; the scripts are `gpu/cpd_ada.py` and `gpu/cpd_diag.py`.
- **Quality and benchmarks:**
  - Table 3 is restructured as models x execution, with an analog sub-table covering 5 models.
  - Table 4 adds the 7B rows; the full per-task results are in the appendix table.
- **Exactness:** more than 2.8 million coded GEMMs in total, all bit-exact. The 7B GSM8K coded generations are 10/10 token-identical to W8A8. 6,272/6,272 coded attention products are exact on 7B.
- **Privacy:** Table 5 now also covers Qwen2.5-7B.
- **Integrity:** tamperers are corrected on the 7B models (980/980 and 1,120/1,120 GEMMs).
- **Cost:** per-GPU worker timing with a new RTX 5000 Ada column and appendix table. INT8 shows little advantage on that card with PyTorch's kernel.
- **GSM8K:** absolute accuracies are not used as quality measurements. They are far below published results, and there is an unexplained BF16/W8A8 gap. The harness is documented in a new appendix section; diagnostics are pending.
- **Main text:** about one page over the 8-page limit; needs compaction.

## Known gaps
- Models are at most 3B parameters, with workers emulated on one GPU; there is no real-network latency measurement.
- The coordinator's encoder and decoder are unfused PyTorch, 11–61x a local GEMM. A fused kernel is future work.
- W8A16 benchmark numbers use 200 examples per task (not 500).
- Install `inconsolata` and re-enable the commented `\usepackage{inconsolata}`, as the ACL template recommends.
