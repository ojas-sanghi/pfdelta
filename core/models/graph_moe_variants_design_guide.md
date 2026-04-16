# Graph MoE Variants Design Guide
This document explains all new MoE implementations, what architectural features each variant enables, and why those features were chosen.

## 01. `graph_moe_v01_baseline`
Implemented architecture features:
- **router_mode**: `combined`
- **gating_mode**: `topk`
Why this exists:
- Reference sparse case-size-aware MoE baseline.

## 02. `graph_moe_v02_size_only`
Implemented architecture features:
- **router_mode**: `size_only`
- **gating_mode**: `topk`
Why this exists:
- Hardest size-specialization prior for case-size OOD transfer studies.

## 03. `graph_moe_v03_learned_router`
Implemented architecture features:
- **router_mode**: `learned`
- **gating_mode**: `topk`
Why this exists:
- Reference sparse case-size-aware MoE baseline.

## 04. `graph_moe_v04_dual_router`
Implemented architecture features:
- **router_mode**: `dual`
- **gating_mode**: `topk`
Why this exists:
- Two independent routers ensembled to reduce routing variance.

## 05. `graph_moe_v05_dense_gate`
Implemented architecture features:
- **router_mode**: `combined`
- **gating_mode**: `dense`
Why this exists:
- Dense expert mixture baseline to compare against sparse top-k modes.

## 06. `graph_moe_v06_noisy_topk`
Implemented architecture features:
- **router_mode**: `combined`
- **gating_mode**: `topk`
- **noisy_std**: `0.15`
Why this exists:
- Noisy top-k routing for anti-collapse and expert exploration.

## 07. `graph_moe_v07_gumbel`
Implemented architecture features:
- **router_mode**: `combined`
- **gating_mode**: `gumbel`
- **temperature**: `0.8`
Why this exists:
- Stochastic sparse-ish routing via Gumbel-Softmax for exploration.

## 08. `graph_moe_v08_hierarchical`
Implemented architecture features:
- **router_mode**: `combined`
- **gating_mode**: `topk`
- **hierarchical**: `True`
Why this exists:
- Two-stage routing: coarse cluster then fine expert assignment.

## 09. `graph_moe_v09_hop_mixed`
Implemented architecture features:
- **router_mode**: `combined`
- **gating_mode**: `topk`
- **expert_kinds**: `['graphconv', 'tag', 'sage', 'tag']`
Why this exists:
- GMoE-style expert diversity through mixed aggregation operators/hop biases.

## 10. `graph_moe_v10_tag_only`
Implemented architecture features:
- **router_mode**: `combined`
- **gating_mode**: `topk`
- **expert_kinds**: `['tag', 'tag', 'tag', 'tag']`
- **tag_k**: `3`
Why this exists:
- GMoE-style expert diversity through mixed aggregation operators/hop biases.

## 11. `graph_moe_v11_sage_only`
Implemented architecture features:
- **router_mode**: `combined`
- **gating_mode**: `topk`
- **expert_kinds**: `['sage', 'sage', 'sage', 'sage']`
Why this exists:
- GMoE-style expert diversity through mixed aggregation operators/hop biases.

## 12. `graph_moe_v12_weak_strong`
Implemented architecture features:
- **router_mode**: `combined`
- **gating_mode**: `topk`
- **weak_strong**: `True`
Why this exists:
- Mowst-inspired weak/strong collaboration (MLP + graph experts).

## 13. `graph_moe_v13_confidence_gate`
Implemented architecture features:
- **router_mode**: `combined`
- **gating_mode**: `topk`
- **weak_strong**: `True`
- **confidence_gate**: `True`
Why this exists:
- Mowst-inspired weak/strong collaboration (MLP + graph experts).
- Confidence-based gating from weak-expert dispersion/entropy.

## 14. `graph_moe_v14_size_temp_low`
Implemented architecture features:
- **router_mode**: `combined`
- **gating_mode**: `topk`
- **temperature**: `0.6`
- **size_prior_strength**: `0.5`
Why this exists:
- Reference sparse case-size-aware MoE baseline.

## 15. `graph_moe_v15_size_temp_high`
Implemented architecture features:
- **router_mode**: `combined`
- **gating_mode**: `topk`
- **temperature**: `1.6`
- **size_prior_strength**: `0.2`
Why this exists:
- Reference sparse case-size-aware MoE baseline.

## 16. `graph_moe_v16_sparse_k1`
Implemented architecture features:
- **router_mode**: `combined`
- **gating_mode**: `topk`
- **top_k**: `1`
Why this exists:
- Reference sparse case-size-aware MoE baseline.

## 17. `graph_moe_v17_sparse_k3`
Implemented architecture features:
- **router_mode**: `combined`
- **gating_mode**: `topk`
- **top_k**: `3`
Why this exists:
- Reference sparse case-size-aware MoE baseline.

## 18. `graph_moe_v18_many_experts`
Implemented architecture features:
- **router_mode**: `combined`
- **gating_mode**: `topk`
- **n_experts**: `6`
- **top_k**: `2`
- **expert_kinds**: `['graphconv', 'sage', 'tag', 'graphconv', 'tag', 'sage']`
Why this exists:
- GMoE-style expert diversity through mixed aggregation operators/hop biases.

## 19. `graph_moe_v19_hier_dual`
Implemented architecture features:
- **router_mode**: `dual`
- **gating_mode**: `topk`
- **hierarchical**: `True`
Why this exists:
- Two-stage routing: coarse cluster then fine expert assignment.
- Two independent routers ensembled to reduce routing variance.

## 20. `graph_moe_v20_hier_gumbel`
Implemented architecture features:
- **router_mode**: `combined`
- **gating_mode**: `gumbel`
- **hierarchical**: `True`
- **temperature**: `0.9`
Why this exists:
- Two-stage routing: coarse cluster then fine expert assignment.
- Stochastic sparse-ish routing via Gumbel-Softmax for exploration.

## 21. `graph_moe_v21_conf_hier`
Implemented architecture features:
- **router_mode**: `combined`
- **gating_mode**: `topk`
- **hierarchical**: `True`
- **weak_strong**: `True`
- **confidence_gate**: `True`
Why this exists:
- Mowst-inspired weak/strong collaboration (MLP + graph experts).
- Confidence-based gating from weak-expert dispersion/entropy.
- Two-stage routing: coarse cluster then fine expert assignment.

## 22. `graph_moe_v22_dual_noisy`
Implemented architecture features:
- **router_mode**: `dual`
- **gating_mode**: `topk`
- **noisy_std**: `0.1`
Why this exists:
- Two independent routers ensembled to reduce routing variance.
- Noisy top-k routing for anti-collapse and expert exploration.

## 23. `graph_moe_v23_dense_weak`
Implemented architecture features:
- **router_mode**: `combined`
- **gating_mode**: `dense`
- **weak_strong**: `True`
- **confidence_gate**: `True`
Why this exists:
- Mowst-inspired weak/strong collaboration (MLP + graph experts).
- Confidence-based gating from weak-expert dispersion/entropy.
- Dense expert mixture baseline to compare against sparse top-k modes.

## 24. `graph_moe_v24_all_features_combo`
Implemented architecture features:
- **router_mode**: `dual`
- **gating_mode**: `gumbel`
- **hierarchical**: `True`
- **weak_strong**: `True`
- **confidence_gate**: `True`
- **noisy_std**: `0.1`
- **temperature**: `0.85`
- **expert_kinds**: `['graphconv', 'tag', 'sage', 'tag']`
Why this exists:
- Mowst-inspired weak/strong collaboration (MLP + graph experts).
- Confidence-based gating from weak-expert dispersion/entropy.
- GMoE-style expert diversity through mixed aggregation operators/hop biases.
- Two-stage routing: coarse cluster then fine expert assignment.
- Stochastic sparse-ish routing via Gumbel-Softmax for exploration.
- Two independent routers ensembled to reduce routing variance.
- Noisy top-k routing for anti-collapse and expert exploration.

## Mapping to provided papers
- **OpenReview K9xHDD6mic (GMoE)** influenced sparse top-k routing, load balancing, and expert diversity via mixed hop/operator experts.
- **OpenReview wYvuY60SdD (Mowst)** influenced weak-vs-strong expert decomposition and confidence-based mixing logic.
- **arXiv 2511.08610 (MoE-GraphSAGE for power stability)** influenced domain framing: graph-expert routing for power-system operating modes and case-scale specialization.
