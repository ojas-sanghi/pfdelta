# MoE Variant Registry Index

This folder contains separate registered model implementations for graph MoE experiments.

## Implemented variants
- graph_moe_v01_size_topk
- graph_moe_v02_size_dense
- graph_moe_v03_noisy_topk
- graph_moe_v04_expert_dropout
- graph_moe_v05_nodes_only_router
- graph_moe_v06_density_router
- graph_moe_v07_no_prior
- graph_moe_v08_strong_prior
- graph_moe_v09_sage_experts
- graph_moe_v10_tag_experts
- graph_moe_v11_nnconv_experts
- graph_moe_v12_hybrid_graphconv_tag
- graph_moe_v13_hybrid_sage_nnconv
- graph_moe_v14_adaptive_topk
- graph_moe_v15_graph_context
- graph_moe_v16_confidence_weak_strong
- graph_moe_v17_confidence_noisy
- graph_moe_v18_confidence_hybrid
- graph_moe_v19_dense_confidence
- graph_moe_v20_adaptive_confidence
- graph_moe_v21_tag_dense
- graph_moe_v22_sage_noisy_adaptive
- graph_moe_v23_nnconv_dropout_adaptive
- graph_moe_v24_full_combo

Each file registers exactly one model name via `@registry.register_model(...)`.
