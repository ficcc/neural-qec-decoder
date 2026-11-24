# Neural pre-decoder + matching — Design

**Date:** 2026-11-24 · **Author:** Uhenraj Jarga Magar · **Status:** draft
**Builds on:** `2026-11-20-neural-qec-decoder-design.md` (noise model, scenarios, evaluation harness, run recorder)

## 1. Intent and why

The end-to-end GNN decoder ties matching at d = 3 but is ~2× worse at d = 7, does not transfer from d = 3,5 to d = 7, and warm-starting it recovers only ~5% (E3, E8). A global neural decoder is therefore the wrong tool at larger distance. Matching is accurate, fast and scales, but cannot use learned context.

**Question:** can a *local* neural pre-decoder remove the easy faults and hand a smaller residual syndrome to matching, with (a) logical error rate no worse than matching alone, (b) the same trained weights working at a larger distance than they were trained on, and (c) a measurably smaller problem for matching?

Success is judged on the three together. A pre-decoder that is accurate but removes almost nothing, or removes a lot but costs accuracy, is a negative or partial result and is reported as such.

## 2. Goals, non-goals, anti-goals

**Goals**
- A pre-decoder trained at d ∈ {3, 5} on S0 (uniform SI1000), evaluated **zero-shot at d = 7** with the same weights and the same confidence threshold.
- End-to-end logical error rate (LER) of `predecoder + matching` vs matching alone, on identical shots, at p ∈ {0.002 … 0.006}, d ∈ {3, 5, 7}.
- A trade-off sweep over the confidence threshold τ: LER ratio vs fraction of fired detectors removed.
- Removal precision: of the pairs/boundary defects removed, the fraction that were truly isolated faults.
- Decode-time of matching on the residual vs on the full syndrome (C++ path), reported separately from the pre-decoder's own inference time.
- Reference: a pre-decoder trained at d = 7, to separate "does not transfer" from "does not work".
- Secondary (only if the primary finishes): S4 crosstalk and S2.

**Non-goals**
- Per-shot reweighting of matching edges (rejected: 100–1000× slower, see §8).
- Hardware-grade latency; real hardware data; leakage.
- Replacing matching: matching always decodes the residual.
- New noise models.

**Anti-goals** (failure even if the code runs)
- Choosing τ on the test shots, or choosing it differently at d = 7. τ is fixed on a **validation set at d = 5** and applied unchanged.
- Comparing against a weaker matching decoder. The residual is decoded with the same graph and weights as the baseline.
- Reporting the speed-up without the pre-decoder's own inference time next to it.
- Claiming transfer from a single seed. Report mean ± std over 3 training seeds.

## 3. Constraints

- Reuses `src/graph.py` (fired-detector kNN graphs, k = 8), the message-passing blocks in `src/gnn.py`, `src/dem_tools.py` (matching graphs with correct logical labels), `src/evaluate.py`, `src/runlog.py`.
- Plain PyTorch, no PyTorch Geometric. Python 3.12, existing venv.
- Compute: training on a rented RTX 4090 (≈ $0.3–0.6) or locally on CPU (a few hours); evaluation on the local PC. **Any rented GPU needs explicit approval of its hourly price first.**
- Commits: authored by the project author only.
- Every run is recorded with `scripts/record_run.py` so it can be compared with earlier runs.

## 4. Method

### 4.1 What the pre-decoder predicts

Matching's graph has one edge per single fault (two detectors, or one detector and the boundary), each with a logical-flip label. A fault is **isolated** in a shot when it fired and no other fired fault touches either of its detectors. An isolated fault is the unambiguous explanation of its detectors, so removing it is safe.

The pre-decoder is a GNN on the fired-detector graph. It outputs, for each **candidate**, the probability that it is an isolated fault:
- *pair candidates*: undirected pairs (u, v) of fired detectors joined by a kNN edge **and** by an edge of the matching graph;
- *boundary candidates*: a fired detector u that has a boundary edge in the matching graph.

Model: node/edge encoders and message-passing blocks as in `GNNDecoder` (hidden 64, 4 layers), then a **symmetric edge head** (input: h_u + h_v, |h_u − h_v|, edge features) for pairs and a node head for boundary candidates. Same distance-independent node features as the end-to-end GNN.

### 4.2 Labels (from the fault-level sampler)

`stim.DetectorErrorModel.compile_sampler().sample(shots, return_errors=True)` returns, per shot, which DEM mechanisms fired. Each mechanism is mapped (once, per distance) to the matching-graph edges it contributes (its decomposed components). Per shot: parity of fired components per edge, count of fired edges touching each detector, and the isolated-fault indicator per edge. Candidate label = 1 if that edge is an isolated fault. Computation is done from the sparse list of fired mechanisms, not dense matrices.

### 4.3 Inference

1. Run the pre-decoder; keep candidates with probability ≥ τ.
2. Resolve conflicts greedily by descending probability: each detector may be used by at most one removal.
3. Residual syndrome = syndrome XOR the removed detectors. Logical contribution = XOR of the removed edges' logical labels.
4. Decode the residual with static matching (same graph and weights as the baseline). Prediction = matching's prediction XOR the logical contribution.

### 4.4 Training

- On-the-fly shots from the sampler, mixed over d ∈ {3, 5} and a band of p (E1 grid, 0.0015–0.010).
- Binary cross-entropy with a positive-class weight for imbalance; 12 000 steps × 1 024 shots, cosine LR, early stopping on a fixed validation set — same recipe as the end-to-end GNN, so results are comparable.
- Seeds, validation and test ranges from `src/seeds.py`; they never overlap.
- 3 training seeds for the d ∈ {3,5} model; 1–3 for the d = 7 reference.

## 5. Experiments (E9)

| ID | What | Output |
|---|---|---|
| E9a | Label sanity: the fraction of fired detectors that are in isolated faults, by d and p; labels reproduce the syndrome | table, test |
| E9b | τ chosen on a d = 5 validation set to maximise removal subject to LER ratio ≤ 1.02 | the single τ used everywhere |
| E9c | LER of matching vs predecoder+matching at d = 3, 5, 7 (d = 7 zero-shot), p = 0.002…0.006, S0 | table and figure, mean ± std over seeds |
| E9d | τ sweep at d = 7: LER ratio vs fraction of detectors removed | trade-off figure |
| E9e | Removal precision and recall of isolated faults by d | table |
| E9f | Timing: matching on residual vs full, plus pre-decoder inference | table |
| E9g | Reference: pre-decoder trained at d = 7 | table |

**Decision rule (stated now, before any test run):** the primary result is *positive* if at d = 7 zero-shot the LER ratio (pre-decoder+matching ÷ matching) is ≤ 1.05 at the validation-chosen τ **and** at least 30% of fired detectors are removed at p = 0.002. It is *partial* if only one holds, and *negative* otherwise. A ratio below 1 is reported as a gain only if the paired difference is significant (per-shot results are stored for a paired test).

## 6. Statistics and recording

- Per-shot error indicators for every decoder are stored (bit arrays), so paired comparisons on identical shots are possible. This also fixes the earlier gap where overlapping intervals could not be resolved.
- Wilson intervals as before; ≥ 100 oracle errors per point.
- A run record (`runs/<label>/`) for the pre-decoder results with environment, commit, configs and test results; compare against baseline v1/v2 with `scripts/compare_runs.py`.

## 7. Repo layout (additions)

```
src/dem_labels.py      mechanism -> matching-edge map; isolated-fault labels from the fault sampler
src/predecoder.py      model, candidate construction, inference with conflict resolution
src/train_pd.py        training loop (reuses config/seed conventions of train.py)
experiments/e9_predecoder.py   evaluation, tau selection, sweeps, plots, --shard/--plot like E1-E8
tests/                 label correctness, candidate construction, conflict resolution, end-to-end invariants
```

## 8. Approaches considered

- **Pre-decoder + matching (chosen).** Local, so it can transfer across distance; uses existing graph code; shrinks matching's problem. Risk: a wrong removal cannot be undone, hence the threshold.
- **Neural edge reweighting, per shot (rejected).** PyMatching has no per-shot weight API. Measured: ≈ 0.5 ms per shot for 50 edge updates at d = 5, ≈ 5–50 ms for all edges, against 6–20 µs for static decoding: 100–1000× slower, and evaluation would be ~10 min per test point.
- **Stacking the existing GNN with matching (deferred).** Cheap, no new training, but at d = 7 the GNN is 2× worse so it would mostly copy matching. May be added later as a comparison.

## 9. Invariants the tests must pin

- With τ = 1 (nothing removed) the pipeline reproduces matching's predictions exactly.
- With oracle probabilities (the true isolated-fault labels) the pre-decoder+matching LER is not worse than matching's on the same shots.
- Removing a pair always flips exactly its two detectors in the residual; no detector is used twice.
- Isolated-fault labels XOR with the remaining fired faults reproduce the observed syndrome.
- Candidates are identical regardless of the order of shots in a batch; an empty or single-defect syndrome works.

## 10. Left to implementation

- The positive-class weight; k for the kNN graph if recall of true pairs is low (measure first); optimizer settings; plot styling.
- Whether to add a second training distance mix if d = 5 alone transfers poorly (report both).
- Whether S4/S2 are run (secondary, only after the primary and only after approval of any further GPU spend).

## 11. Confirmatory test E9b — pre-registered 2026-11-24, after the E9 results were known

**Disclosure.** E9's pre-registered verdict was *partial*: at d = 7 zero-shot, p = 0.002 the pre-decoder removed 88% of fired detectors but had 1.46× matching's logical error rate. A post-hoc sweep of the threshold on the E9 test shots suggested that a stricter τ (≥ 0.95) would meet both bars. E9b tests that hypothesis. The *rule* below was designed after seeing that exploratory result, so a positive outcome supports the hypothesis but is not independent of that history. τ is chosen and tested on data that were not used before.

**Models:** the same four trained models; no retraining.

**New threshold rule.** On **fresh** d = 5 validation shots (circuit sampler, `val_seed(600..602)`, 100 000 shots at each of p = 0.002, 0.004, 0.006, paired with matching on the same shots), for each τ in the grid and each p compute the ratio pre-decoder-errors ÷ matching-errors (pooled over the three d ∈ {3,5}-trained seeds) and the pooled removed fraction. **τ′ = the τ with the largest removed fraction among those whose ratio is ≤ 1.02 at every p**; if none qualifies, the largest τ in the grid. The per-τ, per-p table and τ′ are written to `results/e9b_tau.json` and committed **before any confirmatory test shot is run**.

**Test.** Fresh test shots (`test_seed` chunks from offset 1000, disjoint from every seed used so far), d ∈ {3, 5, 7}, p ∈ {0.002 … 0.006}, scenario S0, the same error budget as E9 (≥ 1 000 matching errors at d = 7, p = 0.002). Decoders: matching and `pd_s0..2`, `pd_ref`, all at τ′. No threshold sweep.

**Confirmation criterion (the same rule as E9):** `classify_result(ratio, removed_fraction)` at d = 7, p = 0.002 with ratio = mean over the three d ∈ {3,5}-trained seeds. *Positive* → the hypothesis is confirmed; *partial* or *negative* → not confirmed. Everything else (ratios at all d and p, paired tests, whether the in-distribution gains at d = 3, 5 survive the stricter τ′) is descriptive.

**No other changes.** Any deviation from this section will be reported as one.
