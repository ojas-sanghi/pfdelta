# PFDelta Graph-MoE Implementation Catalog (24 Variants)

This document explains the newly added **dozens of Mixture-of-Experts implementations** for PFDelta, how they relate to the three papers you linked, and why each architectural feature exists.

## Papers mapped to implementations

## A) MoE-GraphSAGE-based integrated evaluation (arXiv:2511.08610)
Core ideas used:
- GraphSAGE-family message passing for power-system graph dynamics.
- Multi-expert decomposition of operating/stability regimes.
- Router-driven specialization over node and graph conditions.

Implemented in this code drop via:
- GraphSAGE-inclusive expert mixtures (`v08`, `v18`, `v23`, `v24`).
- Case-size-aware routing priors across all GenericGraphMoE variants.

## B) Graph Mixture of Experts / GMoE (OpenReview K9xHDD6mic)
Core ideas used:
- Sparse top-k gating.
- Expert diversity via different aggregation scopes/backbones.
- Load balancing and z-style regularization diagnostics.

Implemented in this code drop via:
- Sparse top-k and switch style (`v01`, `v09`, `v16`).
- Dense MoE control (`v03`) and hard routing control (`v02`, `v15`).
- Hop/backbone-diverse experts (`v07`, `v08`, `v23`, `v24`).

## C) Mowst: Mixture of weak & strong experts (OpenReview wYvuY60SdD)
Core ideas used:
- Weak expert (MLP) + strong expert (GNN).
- Confidence-based gating from weak expert dispersion.
- Learned-vs-manual confidence options.

Implemented in this code drop via:
- Manual confidence gate: `v21`.
- Learned confidence gate: `v22`.

---

## Design principles for OOD generalization on PFDelta

1. **Case-size routing**: all GenericGraphMoE variants can use graph size signals (nodes/edges/density) and a learned case-size prior.
2. **Sparse conditional computation**: top-k or hard switch reduces expert interference and encourages specialization.
3. **Diversity among experts**: some variants mix GraphConv/SAGE/GCN/TAG/GAT experts.
4. **Weak-strong cooperation**: Mowst-inspired variants route easy nodes to weak MLP and harder nodes to strong GNN.
5. **Combination variants**: some variants stack multiple ideas at once to test synergistic gains.

---

## Variant index

| ID | Model name | MoE style | Key features |
|---:|---|---|---|
| 01 | `graph_moe_v01_baseline` | Sparse top-k | Case-size prior + top-k router |
| 02 | `graph_moe_v02_top1_hard` | Hard switch | Single expert per node |
| 03 | `graph_moe_v03_dense` | Dense MoE | Weighted blend over all experts |
| 04 | `graph_moe_v04_noisy` | Sparse top-k | Noisy router logits for exploration |
| 05 | `graph_moe_v05_prior_heavy` | Sparse top-k | Strong case-size prior influence |
| 06 | `graph_moe_v06_prior_free` | Sparse top-k | Pure learned routing, no size prior |
| 07 | `graph_moe_v07_hop_mix` | Sparse top-k | GraphConv + TAG mixed experts |
| 08 | `graph_moe_v08_gat_mix` | Sparse top-k | GraphConv/SAGE/GCN/GAT experts |
| 09 | `graph_moe_v09_switch_sparse` | Switch | top-1 sparse routing |
| 10 | `graph_moe_v10_router_no_size` | Sparse top-k | Router uses node reps only |
| 11 | `graph_moe_v11_many_experts` | Sparse top-k | 8 experts for capacity scaling |
| 12 | `graph_moe_v12_deep_experts` | Sparse top-k | Deeper expert towers |
| 13 | `graph_moe_v13_wide_router` | Sparse top-k | Larger router MLP |
| 14 | `graph_moe_v14_small_router` | Sparse top-k | Compact router MLP |
| 15 | `graph_moe_v15_case_bucket` | Hard + prior | Near-bucketized case routing |
| 16 | `graph_moe_v16_balanced_sparse` | Sparse top-k | Higher-k + mild noise for load spread |
| 17 | `graph_moe_v17_graphconv_only` | Sparse top-k | Homogeneous GraphConv experts |
| 18 | `graph_moe_v18_sage_only` | Sparse top-k | Homogeneous GraphSAGE experts |
| 19 | `graph_moe_v19_gcn_only` | Sparse top-k | Homogeneous GCN experts |
| 20 | `graph_moe_v20_tag_only` | Sparse top-k | Homogeneous TAG experts |
| 21 | `graph_moe_v21_mowst_manual` | Weak/Strong | Manual confidence gate |
| 22 | `graph_moe_v22_mowst_learned` | Weak/Strong | Learned confidence gate net |
| 23 | `graph_moe_v23_combo_noisy_hopmix` | Combo | Noisy + mixed experts + stronger prior |
| 24 | `graph_moe_v24_combo_dense_hybrid` | Combo | Dense routing + hybrid backbones |

---

## File map

### Shared implementation base
- `core/models/moe_variants/base_components.py`

### 24 model files
- `core/models/moe_variants/graph_moe_v01_baseline.py`
- `core/models/moe_variants/graph_moe_v02_top1_hard.py`
- `core/models/moe_variants/graph_moe_v03_dense.py`
- `core/models/moe_variants/graph_moe_v04_noisy.py`
- `core/models/moe_variants/graph_moe_v05_prior_heavy.py`
- `core/models/moe_variants/graph_moe_v06_prior_free.py`
- `core/models/moe_variants/graph_moe_v07_hop_mix.py`
- `core/models/moe_variants/graph_moe_v08_gat_mix.py`
- `core/models/moe_variants/graph_moe_v09_switch_sparse.py`
- `core/models/moe_variants/graph_moe_v10_router_no_size.py`
- `core/models/moe_variants/graph_moe_v11_many_experts.py`
- `core/models/moe_variants/graph_moe_v12_deep_experts.py`
- `core/models/moe_variants/graph_moe_v13_wide_router.py`
- `core/models/moe_variants/graph_moe_v14_small_router.py`
- `core/models/moe_variants/graph_moe_v15_case_bucket.py`
- `core/models/moe_variants/graph_moe_v16_balanced_sparse.py`
- `core/models/moe_variants/graph_moe_v17_graphconv_only.py`
- `core/models/moe_variants/graph_moe_v18_sage_only.py`
- `core/models/moe_variants/graph_moe_v19_gcn_only.py`
- `core/models/moe_variants/graph_moe_v20_tag_only.py`
- `core/models/moe_variants/graph_moe_v21_mowst_manual.py`
- `core/models/moe_variants/graph_moe_v22_mowst_learned.py`
- `core/models/moe_variants/graph_moe_v23_combo_noisy_hopmix.py`
- `core/models/moe_variants/graph_moe_v24_combo_dense_hybrid.py`

### Configs (25 total)
- `core/configs/ojas_configs/moe_variants/master_moe_variants.yaml`
- plus one YAML per variant in `core/configs/ojas_configs/moe_variants/graph_moe_v*.yaml`

---

## Experimental guidance

1. Start with `v01`, `v05`, `v06`, `v09`, `v21`, `v22`.
2. Compare sparse vs dense (`v01` vs `v03`).
3. Compare prior-heavy vs prior-free (`v05` vs `v06`) to isolate case-size prior effect.
4. Compare homogeneous vs heterogeneous experts (`v17` vs `v08`/`v23`).
5. Compare weak/strong confidence variants (`v21` vs `v22`).

Use routing diagnostics (`model.diagnostics`) to analyze specialization and collapse.
