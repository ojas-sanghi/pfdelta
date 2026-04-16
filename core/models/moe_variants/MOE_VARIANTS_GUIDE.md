# PFDelta Graph-MoE Variant Suite (Detailed Guide)

This directory provides a broad set of graph Mixture-of-Experts (MoE) implementations so we can run many controlled experiments for out-of-distribution (OOD) generalization across PFDelta case sizes.

## Why so many variants?

You asked for separate pathways and combinations. Instead of a single architecture, this suite includes multiple independent model files, each registered with a different `model.name`, so they can be launched as separate experiments.

---

## Paper-informed design mapping

## 1) `https://openreview.net/pdf?id=K9xHDD6mic` (Graph Mixture of Experts)
Main ideas reflected in this suite:
- sparse top-k routing,
- routing at node level,
- experts with diverse aggregation scopes/hops,
- load/expert utilization diagnostics,
- noisy gating option.

Implemented in:
- `graph_moe_casesize.py`
- `graph_moe_noisy_topk.py`
- `graph_moe_switch.py`
- `graph_moe_hophybrid.py`
- `graph_moe_multiscale_combo.py`

## 2) `https://openreview.net/pdf?id=wYvuY60SdD` (Mowst: weak + strong experts)
Main ideas reflected in this suite:
- weak expert (MLP),
- strong expert (GNN),
- confidence-style gating from weak-expert uncertainty/dispersion.

Implemented in:
- `graph_moe_weak_strong.py`

## 3) `https://arxiv.org/pdf/2511.08610` (power-system MoE GraphSAGE framing)
Main ideas reflected in this suite:
- application-specific MoE for power-system settings,
- explicit specialization pressure for grid-operating regimes,
- unified model with expert routing.

Implemented via case-size-oriented routing in:
- `graph_moe.py` (existing root MoE file)
- `graph_moe_casesize.py`
- `graph_moe_casebins.py`
- `graph_moe_prioronly.py`

---

## Variant catalog

| Model name (`model.name`) | File | Key MoE feature(s) | Why it matters |
|---|---|---|---|
| `graph_moe` | `core/models/graph_moe.py` | case-size prior + sparse top-k + aux stats | baseline full-feature case-size MoE |
| `graph_moe_casesize` | `graph_moe_casesize.py` | sparse top-k + size-conditioned router | direct OOD-by-size specialization |
| `graph_moe_hophybrid` | `graph_moe_hophybrid.py` | experts differ by hop radius (`TAGConv K`) | adapts to locality vs long-range dependence |
| `graph_moe_weak_strong` | `graph_moe_weak_strong.py` | weak/strong expert with confidence gate | Mowst-style calibrated compute allocation |
| `graph_moe_switch` | `graph_moe_switch.py` | hard top-1 routing (Switch) | maximal sparsity, efficiency stress test |
| `graph_moe_dense` | `graph_moe_dense.py` | dense soft routing over all experts | no hard routing discontinuity |
| `graph_moe_noisy_topk` | `graph_moe_noisy_topk.py` | noisy top-k gating | anti-collapse + exploration |
| `graph_moe_hierarchical` | `graph_moe_hierarchical.py` | graph-level group router + node router | coarse-to-fine specialization |
| `graph_moe_attention_router` | `graph_moe_attention_router.py` | attention-based router | richer conditional routing context |
| `graph_moe_residual` | `graph_moe_residual.py` | MoE output + residual skip | stability + optimization support |
| `graph_moe_multiscale_combo` | `graph_moe_multiscale_combo.py` | mixed expert families + size prior + noisy top-k | combined pathway of multiple ideas |
| `graph_moe_prioronly` | `graph_moe_prioronly.py` | deterministic size prior routing only | isolates effect of hand-crafted prior |
| `graph_moe_dual_router` | `graph_moe_dual_router.py` | semantic + structural dual routers | disentangles feature vs structure routing |
| `graph_moe_dropout_experts` | `graph_moe_dropout_experts.py` | expert dropout during routing | robustness + anti-over-reliance |
| `graph_moe_casebins` | `graph_moe_casebins.py` | hard size-bin assignment | explicit expert-per-size policy |

---

## Combination pathways explicitly included

1. **Case size + sparse routing**: `graph_moe_casesize`
2. **Case size + noisy sparse routing**: `graph_moe_noisy_topk`
3. **Case size + hop-diverse experts**: `graph_moe_hophybrid`
4. **Case size + mixed family experts + noisy + prior**: `graph_moe_multiscale_combo`
5. **Case size + hard bins**: `graph_moe_casebins`
6. **Weak/strong confidence routing**: `graph_moe_weak_strong`
7. **Hierarchical coarse/fine routing**: `graph_moe_hierarchical`

---

## Configuration files

A dedicated batch config has been created for each variant under:

`core/configs/ojas_configs/moe_suite/`

Each config is `simple_batch` style and follows existing Ojas config patterns:
- train + val PFDelta PFNet datasets,
- model-specific `model.name` and model arguments,
- shared PF losses (MSE, masked MSE, universal power balance),
- seed and LR sweeps.

This means each implementation can be launched independently from `main.py` by selecting the corresponding config path.

---

## Recommended execution order

1. Start with:
   - `graph_moe_task13_batch.yaml`
   - `graph_moe_casesize_task13_batch.yaml`
2. Compare sparse routing families:
   - `graph_moe_switch_task13_batch.yaml`
   - `graph_moe_noisy_topk_task13_batch.yaml`
   - `graph_moe_dense_task13_batch.yaml`
3. Compare structural diversity experts:
   - `graph_moe_hophybrid_task13_batch.yaml`
   - `graph_moe_multiscale_combo_task13_batch.yaml`
4. Evaluate weak/strong paradigm:
   - `graph_moe_weak_strong_task13_batch.yaml`
5. Stress-test priors and hard assignment:
   - `graph_moe_prioronly_task13_batch.yaml`
   - `graph_moe_casebins_task13_batch.yaml`

---

## Notes

- All implementations are intentionally in separate files to make ablations explicit and reproducible.
- Most variants preserve the same forward signature (`forward(data) -> Tensor`) to remain compatible with existing trainers/losses.
- Routing diagnostics are exposed on model attributes where relevant for analysis scripts.
