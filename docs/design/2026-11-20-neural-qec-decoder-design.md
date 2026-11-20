# Neural QEC Decoder under Mismatched Noise — Design

**Date:** 2026-11-20 · **Author:** Uhenraj Jarga Magar · **Status:** draft

## 1. Intent and why

This is the second portfolio project for PhD applications to Monash (Scalable Quantum Systems Lab) and UTS (Centre for Quantum Software and Information). Both list AI-assisted QEC decoding as a research theme. The first project (`hybrid-qnn-nisq-benchmark`) showed the author can run rigorous, honest QML benchmarks. This one shows the same rigour on a problem where the metric is objective, the **logical error rate (LER)**.

**Research question:** a minimum-weight perfect matching (MWPM) decoder is only as good as its noise model. When the real device's noise differs from the model the decoder was given (uneven per-qubit error rates, a few bad couplers, drift since the last calibration), how much does a graph neural network (GNN) decoder trained only on syndrome data recover, compared with matching given naive, calibrated and exact noise models? And does the GNN still work at larger code distances than it was trained on?

**Control:** under uniform noise all three matching variants coincide, so any GNN gain there comes purely from exploiting correlations that matching discards (a Y error triggers both X- and Z-type detectors, a hyperedge that matching must split in two). That separates "learns the noise" from "learns correlations".

## 2. Goals, non-goals, anti-goals

**Goals**
- Rotated surface-code memory experiments (Z basis) at d = 3, 5, 7 with rounds = d, under circuit-level noise.
- Four noise scenarios (§4) and four decoders (§5), with LER versus physical error rate *p* for each.
- A threshold plot for the uniform-noise control.
- A distance-generalisation test: GNN trained on d ≤ 5, evaluated at d = 7, against a GNN trained at d = 7.
- Honest latency numbers for the GNN and MWPM, per shot and per round.
- Multi-seed statistics: mean ± std over training seeds *and* over sampled noise realisations ("device instances").
- A workshop-style README in the same style as the first project.

**Non-goals**
- Leakage, erasure, or non-Pauli noise (Stim cannot simulate them, and they're out of scope for 2–3 weeks).
- Real hardware data, or FPGA/ASIC real-time decoding.
- Other codes (colour, qLDPC), lattice surgery, or logical gates.
- Transformer or CNN decoders.
- X-basis memory (a stretch goal only, in §9).

**Anti-goals** (failure even if the code runs)
- Claiming the GNN "beats MWPM" when it only beats *naive* matching. The headline comparison is against **calibrated** matching, with oracle matching as the ceiling.
- Any choice of model, hyperparameter or checkpoint made by looking at the test set.
- Reporting only the device instances where the GNN looks good.
- Implying the latency is hardware-ready. Real hardware needs ~1 µs per round, and a Python GNN on CPU won't come close. Say so.

## 3. Constraints

- **Time:** 2–3 weeks. The visa expires in late March 2027, so polish matters more than breadth.
- **Compute:** local torch is CPU-only (2.15.0+cpu). Develop and run d = 3, 5 locally. Use a RunPod GPU only if d = 7 training is impractically slow on CPU, and record the GPU type and hours used.
- **Environment:** a new sibling repo at `G:\Research\neural-qec-decoder`. Use `uv venv --python 3.12`, because the `py -3.12` launcher is broken on this machine. Run things as `python -m experiments.<name>` from the repo root, mirroring the first project.
- **Dependencies:** stim, pymatching, sinter, torch, numpy, matplotlib. **No PyTorch Geometric** (my call: Windows wheels for this torch version are unreliable). Message passing is written in plain torch with `index_add_` / `scatter_reduce`.
- **Commits:** authored by the project author only.

## 4. Noise model and scenarios

**Base model (revised 2026-11-20 after reviewing the blueprint report):** **SI1000** superconducting-inspired circuit noise:
- DEPOLARIZE2(p) after each CX.
- DEPOLARIZE1(p/10) after single-qubit Cliffords.
- X_ERROR(2p) after resets.
- X_ERROR(5p) before measurements.
- DEPOLARIZE1(p/10) on idle qubits each moment.
- DEPOLARIZE1(2p) on qubits idling during a measurement/reset moment (resonator idle).

The scenario multipliers in the table below apply per channel on top of these SI1000 base rates. The Stim-built-in-compatible structure described next is retained only as the exact-equivalence unit test of the insertion pass.

*Original text, kept for the equivalence test:* circuit-level depolarising noise with the same structure as Stim's built-in generated noise. It is applied by our own noise-insertion pass over a noiseless `stim.Circuit.generated("surface_code:rotated_memory_z", ...)` circuit, so that rates can differ per qubit and per gate:
- DEPOLARIZE2(p_g) after each CX.
- DEPOLARIZE1(p_q) after single-qubit Cliffords.
- DEPOLARIZE1(p_q) on data qubits at the start of each round (this stands in for idle noise).
- X_ERROR(p_q) after resets and before measurements.

*(Design note: explicit per-tick idle noise was dropped in favour of Stim's before-round data depolarisation. This keeps the equivalence test exact, and per-qubit data rates still model idling mismatch.)*

Each gate and qubit has its own rate, derived from a nominal *p*. The insertion pass is unit-tested by checking that with uniform rates it reproduces the detector error model of Stim's built-in generated noise exactly (same error mechanisms and probabilities, to tolerance).

| ID | Scenario | True noise | Role |
|----|----------|-----------|------|
| S0 | Uniform | all rates = *p* | **Control** (correlated-Y only) |
| S1 | Heterogeneous | per-qubit/gate rate = *p*·exp(σ·z), z ~ N(0,1), σ ≈ 0.5, renormalised so the mean = *p* | static mismatch |
| S2 | Heterogeneous + bad couplers | S1, plus k ≈ 2–3 CNOT locations at 5–10× rate | localised mismatch |
| S3 | Drift | S2 rates at time t₀ for calibration/training; at test time t₁, each rate multiplied by exp(δ·z′), δ ≈ 0.3 | stale calibration |

A **device instance** is one random draw of the per-qubit/gate rates (z, the bad-coupler choice, z′), fixed by a seed. Every result for S1–S3 is reported over ≥ 3 device instances.

The exact σ, δ, k and multipliers are implementation choices within the ranges above. They must be fixed before any test-set evaluation and recorded in a config file.

## 5. Decoders

| Decoder | What it knows | Implementation |
|---------|---------------|----------------|
| **Naive MWPM** | uniform model at nominal *p* | PyMatching from the uniform-noise DEM |
| **Calibrated MWPM** | rates estimated from syndrome data | PyMatching with edge weights re-estimated from detection-event correlations (the standard p_ij estimator: p_ij from ⟨x_i x_j⟩, ⟨x_i⟩, ⟨x_j⟩; boundary edges from the residual single-detector rates), using **the same training shots the GNN sees** |
| **Oracle MWPM** | the exact true DEM | PyMatching from the true-noise DEM (ceiling for matching) |
| **GNN** | syndromes + labels only | see below |

Under S3, calibrated MWPM and the GNN both use t₀ data and are tested at t₁, so both are equally stale. The oracle uses the t₁ model.

**Optional stronger baseline:** correlated matching, if the installed PyMatching version supports it, or else belief-matching. It matters mostly for S0, where it is the fair opponent for the correlated-Y claim. If neither is available in time, say so in the README rather than overclaim on S0.

**GNN architecture** (following Lange et al. 2023, "Data-driven decoding of quantum error correcting codes using graph neural networks"):
- **Nodes:** fired detectors only. Features are a one-hot stabiliser type, the distance to each of the 4 spatial boundaries in raw lattice units (clipped at 4), and first-round / last-round flags.

  *(Design note: absolute coordinates normalised by d were replaced with local, unnormalised features. Normalised positions would mean different things at different d, which would undermine distance generalisation.)*
- **Edges:** k-nearest neighbours in space-time (k ≈ 6–10). Edge features are (Δx, Δy, Δt) and the inverse distance.
- **Body:** L message-passing layers (edge-conditioned MLP messages, residual connections, layer norm), then global mean + max pooling, then an MLP head giving one logit (logical Z flip).
- **Distance-agnostic:** the same weights run at any d. That's the reason for choosing a GNN.
- **Empty syndrome** (no detectors fired): predict "no flip" without calling the network.

**Training**
- Shots are generated on the fly from Stim, so there is no fixed dataset and no overfitting to a finite set.
- Training *p* is sampled from the sweep range so one model covers the whole curve.
- The validation set is a separate fixed Stim seed, used for early stopping and model selection. The test set uses another separate seed and is evaluated **once**, after selection.
- One GNN per (scenario, device instance, training-distance set).
- Report the parameter count, training shots and wall-clock time.

## 6. Experiments

| ID | Experiment | Output |
|----|-----------|--------|
| E1 | Control: S0, d = 3/5/7, *p* sweep around threshold, all decoders | LER-vs-*p* curves and threshold estimate |
| E2 | Mismatch: S1, S2, S3 at *p* below threshold, all decoders | LER table + curves, gap to oracle |
| E3 | Distance generalisation: GNN(trained d=3,5) vs GNN(trained d=7) at d=7, in S0 and S1 | LER comparison; a negative result is reported as-is |

*Note on E3 (design check):* a device instance is drawn per qubit, so a d = 7 device is necessarily a *different, unseen* device from any d = 3/5 one. E3 therefore runs on S0, and on S1 trained across many random device instances and tested on fresh d = 7 instances. That makes S1-E3 a test of "generalises to a new, larger device", not "learned this device". It is not run on S2/S3, where the point is learning one specific device.

*Training budget (design check):* one GNN per (scenario, device instance, distance) for E1/E2 means up to 4 × 3 × 3 = 36 models before seeds. To keep this CPU-feasible, E2 uses 3 device instances × 1 training seed. The separate ≥ 3-seed requirement applies to S0 (E1) and to one representative S2 instance, which quantifies training-seed variance.
| E4 | Calibration budget: calibrated MWPM vs GNN as training/calibration shots vary | data-efficiency curve |
| E5 | Latency: per-shot and per-round decode time, batch 1 and batched, CPU (and GPU if used) | table vs the ~1 µs/round hardware budget |

**Statistics**
- Report the LER per shot and the per-round LER ε = ½(1 − (1 − 2P_L)^{1/r}).
- Wilson 95% intervals for each point. Enough test shots to observe ≥ 100 logical errors where feasible, and an upper bound where not.
- GNN: ≥ 3 training seeds. S1–S3: ≥ 3 device instances. Report mean ± std across them.
- MWPM collection may use sinter. GNN evaluation uses a custom loop on the same test shots, so every decoder decodes **identical syndromes**.

### 6.1 Revisions adopted 2026-11-20 (blueprint report review)

1. **Training budget:** about 5M shots per model (5000 steps × 1024). Neural decoder accuracy is reported to saturate at 10⁶–10⁷ shots. This cuts compute about 4×.
2. **SI1000 base noise** (§4).
3. **E2 training error rates include near-threshold values** to avoid label starvation at d = 7 below threshold. The test error rates stay below threshold.
4. **Reporting:**
   - The error-suppression factor Λ, from a log-linear fit of ε_L(d) ∝ Λ^{-(d+1)/2} at fixed p.
   - E5 latency at batch sizes B ∈ {1, 64, 4096}, with median and p99.

## 7. Repo layout (proposal)

```
neural-qec-decoder/
  src/noise.py        # per-qubit/gate noise insertion, scenarios, device instances
  src/circuits.py     # circuit + DEM construction
  src/calibrate.py    # p_ij estimator -> reweighted matching graph
  src/graph.py        # syndrome -> graph batch (kNN, features)
  src/gnn.py          # plain-torch message passing model
  src/train.py        # on-the-fly training loop
  src/evaluate.py     # shared test shots, all decoders, Wilson CIs
  experiments/        # e1_control.py ... e5_latency.py
  configs/            # frozen scenario + training configs
  tests/              # noise-pass equivalence, estimator sanity, graph build
  results/ figures/ README.md
```

## 8. Approaches considered

**Noise target** (decided): mismatched / drifting noise as the main study, with correlated Y errors under uniform noise as the control. Rejected alternatives:
- Biased noise: matching already uses a biased DEM correctly, so there's no real win.
- Leakage: Stim can't simulate it, and a custom simulator is too risky for 2–3 weeks.

**Decoder architecture** (decided): GNN.
- **GNN (chosen):** distance-agnostic, sparse and cheap enough for CPU, handles boundaries naturally.
- **3D CNN:** simplest and fastest, but boundaries and cross-distance transfer need padding hacks.
- **Transformer (AlphaQubit-style):** most accurate at a fixed d, but needs a GPU from day one and is likely to overrun the time budget.

## 9. Left to implementation

- Hyperparameters (k, L, hidden width, learning rate, batch size, training-shot budget), chosen on validation only.
- Exact scenario parameters within §4's ranges.
- The *p* grid.
- Whether to use sinter for MWPM collection.
- Plot styling.

**Stretch, only if E1–E5 are done:** X-basis memory; belief-matching baseline; fine-tuning the GNN on a small t₁ sample in S3, compared against recalibrating MWPM on the same sample.
