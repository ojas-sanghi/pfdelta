# MoE Variant Config Suite

This directory contains batch configs for training/evaluating the MoE implementations.

## Files
- `moe_all_variants_batch.yaml`: runs default + all 24 variants.
- `moe_routing_family_batch.yaml`: routing mechanism ablations.
- `moe_size_prior_family_batch.yaml`: size-prior and router-feature ablations.
- `moe_expert_arch_family_batch.yaml`: expert-architecture family ablations.
- `moe_weak_strong_family_batch.yaml`: weak/strong confidence-gated MoE family.
- `moe_default_vs_baselines_batch.yaml`: lightweight baseline comparison.
- `moe_paper_graph_moe_batch.yaml`: Graph-MoE inspired subset.
- `moe_paper_mowst_batch.yaml`: Mowst-inspired subset.
- `moe_paper_graphsage_power_batch.yaml`: MoE-GraphSAGE-inspired subset.

Run with:
```bash
uv run main.py --config ojas_configs/moe_variants/<config_name_without_yaml>
```
