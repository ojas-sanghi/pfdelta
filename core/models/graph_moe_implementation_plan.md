# Graph MoE Program for PFDelta (Expanded, Multi-Pathway Edition)

This document explains:
1. How the three provided MoE papers map onto PFDelta.
2. Which architecture features were implemented.
3. Why those features matter for OOD generalization.
4. The complete list of newly implemented model variants.
5. How to run batch experiments using the new config files.

---

## A) Paper-to-implementation mapping

## A.1 MoE-GraphSAGE (arXiv:2511.08610)
**Core idea in the paper**: multiple experts with a gating mechanism improve multi-regime stability assessment in power systems; GraphSAGE backbone handles graph neighborhoods.

**Mapped implementation features**:
- SAGE-based experts (`expert_type="sage"`).
- Top-k sparse gating for selective expert usage.
- Case-size-aware routing inputs and optional size prior.
- Hybrid stacks where SAGE experts coexist with others.

**Relevant variants**:
- `graph_moe_v09_sage_experts`
- `graph_moe_v13_hybrid_sage_nnconv`
- `graph_moe_v22_sage_noisy_adaptive`

---

## A.2 Graph Mixture of Experts (OpenReview: K9xHDD6mic)
**Core idea in the paper**: sparse top-k routing, expert specialization, balancing losses, and structural diversity handling.

**Mapped implementation features**:
- Dense vs sparse routing modes.
- Noisy routing for better exploration.
- Load/importance balancing diagnostics.
- Adaptive top-k and mixed expert types.
- Hop-style diversity via TAG experts and hybrid mixes.

**Relevant variants**:
- `graph_moe_v01_size_topk`, `v02_size_dense`, `v03_noisy_topk`, `v14_adaptive_topk`, `v21_tag_dense`, `v24_full_combo`.

---

## A.3 Mixture of Weak & Strong Experts on Graphs (OpenReview: wYvuY60SdD)
**Core idea in the paper**: weak expert (e.g., MLP) + strong expert (GNN), confidence-driven collaboration.

**Mapped implementation features**:
- Optional weak MLP expert.
- Confidence head controlling weak-vs-strong interpolation.
- Confidence + sparse/dense/adaptive combinations.

**Relevant variants**:
- `graph_moe_v16_confidence_weak_strong`
- `graph_moe_v17_confidence_noisy`
- `graph_moe_v18_confidence_hybrid`
- `graph_moe_v19_dense_confidence`
- `graph_moe_v20_adaptive_confidence`
- `graph_moe_v24_full_combo`

---

## B) Why experts specialize by case size

PFDelta’s strongest shift axis is system scale (14 → 2000 buses). Case-size-aware routing is therefore a natural mechanism to improve OOD behavior:
- smaller systems often depend more on local interactions,
- larger systems need broader aggregation and capacity,
- mixed systems benefit from smooth interpolation between experts.

Implemented mechanism:
- Router sees node embeddings + graph-level size signals.
- A learnable size prior encourages size-stratified specialization.
- Sparse top-k mixing avoids hard brittle assignment while preserving specialization.

---

## C) Implemented feature catalog

All features are implemented in `core/models/moe_backbones.py` and then combined into separate registered files under `core/models/moe_variants/`.

### C.1 Expert architecture choices
- GraphConv experts
- SAGE experts
- TAG experts (hop-aware flavor)
- NNConv experts (edge-aware)
- Hybrid alternating experts (primary + secondary type)

### C.2 Routing choices
- `router_mode="topk"` (sparse)
- `router_mode="dense"` (soft full mixing)
- Optional noisy gating
- Optional adaptive top-k
- Optional graph-context logit bias

### C.3 Structural priors
- `size_feature_mode`: nodes-only / log(nodes, edges) / nodes-edges-density
- learnable size centers and temperature
- tunable prior-vs-learned blend

### C.4 Weak-strong gating
- weak MLP expert
- confidence head
- convex interpolation between weak and strong MoE output

### C.5 Stabilization / diagnostics
- routing entropy
- expert importance and load distributions
- load-balance and z-loss tracking
- all stored in `model.routing_info` and `model.router_stats`

---

## D) Full list of implementations (24 + default)

Default baseline:
- `graph_moe` (upgraded default)

Explicit variants:
1. `graph_moe_v01_size_topk`
2. `graph_moe_v02_size_dense`
3. `graph_moe_v03_noisy_topk`
4. `graph_moe_v04_expert_dropout`
5. `graph_moe_v05_nodes_only_router`
6. `graph_moe_v06_density_router`
7. `graph_moe_v07_no_prior`
8. `graph_moe_v08_strong_prior`
9. `graph_moe_v09_sage_experts`
10. `graph_moe_v10_tag_experts`
11. `graph_moe_v11_nnconv_experts`
12. `graph_moe_v12_hybrid_graphconv_tag`
13. `graph_moe_v13_hybrid_sage_nnconv`
14. `graph_moe_v14_adaptive_topk`
15. `graph_moe_v15_graph_context`
16. `graph_moe_v16_confidence_weak_strong`
17. `graph_moe_v17_confidence_noisy`
18. `graph_moe_v18_confidence_hybrid`
19. `graph_moe_v19_dense_confidence`
20. `graph_moe_v20_adaptive_confidence`
21. `graph_moe_v21_tag_dense`
22. `graph_moe_v22_sage_noisy_adaptive`
23. `graph_moe_v23_nnconv_dropout_adaptive`
24. `graph_moe_v24_full_combo`

---

## E) Experimental pathway plan

## E.1 Stage 1: Baselines
- Compare `graph_conv`, `powerflownet`, `graph_moe`.

## E.2 Stage 2: Core routing ablations
- `v01/v02/v03/v04/v14`.

## E.3 Stage 3: Structural priors
- `v05/v06/v07/v08`.

## E.4 Stage 4: Expert architecture families
- `v09/v10/v11/v12/v13/v21/v22/v23`.

## E.5 Stage 5: weak-strong MoE families
- `v16/v17/v18/v19/v20/v24`.

## E.6 Stage 6: Full combination shortlist
- `v24` + best-of-each stage.

---

## F) Configs and how to run

New YAMLs live in:
`core/configs/ojas_configs/moe_variants/`

They include:
- focused batch sweeps by family,
- an all-variants mega-batch,
- a default baseline comparison batch.

Run pattern (from repo root):
```bash
uv run main.py --config ojas_configs/moe_variants/moe_all_variants_batch.yaml
```

---

## G) Notes on quality controls

- All variant files are registered models, so `load_registry()` auto-discovers them.
- Implementations share one tested backbone to reduce drift and bugs.
- Per-variant files remain separate for explicit reproducibility and ablation bookkeeping.
