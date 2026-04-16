# Master Experiment Plan: Case-Size-Centric Graph MoE Program for PFDelta

## Objective

Train one or many MoE models that can generalize across in-distribution and OOD test settings by specializing experts over system-scale regimes while preserving transferable shared representations.

## Core hypothesis

- Power-grid case size (14/30/57/118/500/2000) is a dominant axis of distribution shift.
- Routing that conditions on case-size/statistical graph structure can reduce negative transfer.
- Sparse MoE (top-k or top-1) can improve both capacity and efficiency.

## Planned experiment matrix

### Axis A: Routing sparsity
- Dense: `graph_moe_dense`
- Sparse top-k: `graph_moe_casesize`, `graph_moe_noisy_topk`
- Top-1 switch: `graph_moe_switch`

### Axis B: Expert diversity
- Homogeneous GNN experts: most variants
- Hop-diverse experts: `graph_moe_hophybrid`
- Mixed family experts (GraphConv + TAG + MLP): `graph_moe_multiscale_combo`

### Axis C: Prior strength
- No prior / weak prior: `graph_moe_dense`, `graph_moe_noisy_topk`
- Strong deterministic prior: `graph_moe_prioronly`, `graph_moe_casebins`

### Axis D: Routing architecture
- MLP router: baseline sparse families
- Dual router: `graph_moe_dual_router`
- Attention router: `graph_moe_attention_router`
- Hierarchical router: `graph_moe_hierarchical`

### Axis E: Weak/strong MoE paradigm
- Weak+strong confidence model: `graph_moe_weak_strong`

## Expected outcomes

1. **Best raw OOD transfer** likely from `graph_moe_multiscale_combo` or `graph_moe_hophybrid`.
2. **Best efficiency** likely from `graph_moe_switch`.
3. **Best calibration/stability** likely from `graph_moe_weak_strong` and `graph_moe_residual`.
4. **Best interpretability** likely from `graph_moe_casebins` and `graph_moe_prioronly`.

## Metrics to report

- PF masked MSE (regularized and unregularized)
- Universal power balance losses
- Per-case-size validation curves
- OOD held-out-size performance gaps
- Routing entropy, expert load histogram, and load variance

## OOD protocols

1. Train on (14,30,57,118), test on 500.
2. Train on (30,57,118,500), test on 14.
3. Train on all feasible-N and test near-infeasible N-1/N-2.
4. Train Task 1.3 and evaluate transfer to Task 3.x stress settings.

## Diagnostic protocol

For each run, store:
- router entropy trajectory,
- expert usage by case size,
- confusion of size bucket ↔ selected experts,
- residual error percentile by case.

## Prioritized launch schedule

1. Baselines:
   - `graph_moe_task13_batch.yaml`
   - `graph_moe_casesize_task13_batch.yaml`
2. Sparse routing ablation:
   - `graph_moe_switch_task13_batch.yaml`
   - `graph_moe_noisy_topk_task13_batch.yaml`
3. Expert diversity ablation:
   - `graph_moe_hophybrid_task13_batch.yaml`
   - `graph_moe_multiscale_combo_task13_batch.yaml`
4. Routing architecture ablation:
   - `graph_moe_dual_router_task13_batch.yaml`
   - `graph_moe_attention_router_task13_batch.yaml`
   - `graph_moe_hierarchical_task13_batch.yaml`
5. Robustness:
   - `graph_moe_dropout_experts_task13_batch.yaml`
   - `graph_moe_residual_task13_batch.yaml`
6. Explicit prior controls:
   - `graph_moe_prioronly_task13_batch.yaml`
   - `graph_moe_casebins_task13_batch.yaml`
7. Weak/strong:
   - `graph_moe_weak_strong_task13_batch.yaml`

## Interpretation rubric

- If expert load collapses: increase noise, add balancing coefficient, use dropout experts.
- If OOD improves but ID drops: blend stronger shared stem or increase `top_k`.
- If large-case errors dominate: increase hop-diverse experts and/or increase large-size prior coverage.
- If small-case overfit: lower expert count or raise regularization.

## Deliverables in this commit

- 15 separate MoE model implementations.
- 15 independent `simple_batch` Ojas config files.
- Variant explanation and paper-feature mapping docs.

