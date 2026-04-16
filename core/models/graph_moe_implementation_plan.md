# Graph Mixture-of-Experts (MoE) for PFDelta: Detailed Implementation Plan

## 1) Goal and context

PFDelta already supports single-backbone architectures such as GraphConv, PowerFlowNet, GNS, and CANOS. The proposed addition is a **single model** with multiple experts and a router, so that one training run can learn specialized behaviors across case sizes (14/30/57/118/500/2000 buses) while preserving generalization to OOD combinations of topology and operating point.

The design in `graph_moe.py` follows existing repository conventions:
- Registered with the model registry (`@registry.register_model("graph_moe")`).
- `forward(data)` returns a tensor prediction compatible with existing losses.
- Model internals expose diagnostics in attributes without requiring trainer changes.

---

## 2) What each expert should specialize on

### Primary specialization axis: **case size**

For PFDelta, the strongest domain shift is system scale. Different case sizes induce different:
- graph diameters,
- degree distributions,
- stability/operating regimes,
- coupling between local and long-range effects.

So each expert should learn a partially distinct regime over size.

### Practical expert-role mapping

For `n_experts = 4`, a good initial mapping is:
- **Expert 0**: small systems (14, 30)
- **Expert 1**: medium-small (57)
- **Expert 2**: medium-large (118, 500)
- **Expert 3**: large/extreme (500, 2000)

This mapping is implemented softly (not hard-coded as one-hot assignment) through:
1. a **learned router**, and
2. a **size prior** based on normalized log-node count.

This keeps flexibility while still injecting the intended inductive bias.

---

## 3) Design choices informed by graph-MoE literature

Based on the provided papers, the implementation adopts these principles:

1. **Sparse routing (top-k)** rather than dense averaging.
   - From Graph MoE literature: sparse activation improves efficiency and often generalization.

2. **Routing conditioned on structural scale signals**.
   - Router input includes graph-size features (nodes/edges/log variants), not only node embeddings.

3. **Load-balancing auxiliary objectives**.
   - Avoid router collapse to a single expert.
   - Track both importance and actual sparse load.

4. **Optional noisy routing during training**.
   - Improves exploration and prevents premature routing collapse.

5. **Weak/strong complement idea as future extension**.
   - Current model keeps homogeneous experts (all GraphConv towers).
   - A next version can assign one lightweight expert (MLP/shallow GNN) and several stronger experts.

---

## 4) Architecture blueprint

## 4.1 Inputs
- Node input: `data["bus"].x[:, x_start:x_end]` (default slice `[4:10]`, matching existing graph models).
- Graph connectivity: `data["bus", "branch", "bus"].edge_index`.
- Batch vector: `data["bus"].batch` when mini-batched; fallback to a single-graph batch otherwise.

## 4.2 Shared stem
- Two-layer MLP encoder from input features to `hidden_dim`.
- Produces hidden node representation `h` used by both experts and router.

## 4.3 Experts
- `n_experts` independent GraphConv stacks.
- Each expert has identical shape but independent parameters.
- Each expert outputs `[num_nodes, out_dim]`.

## 4.4 Router
- Build graph-size features per graph:
  - nodes only, or
  - log(nodes), log(edges), or
  - nodes, edges, density.
- Broadcast size features to nodes in that graph.
- Concatenate with node hidden representation.
- MLP outputs expert logits per node.

## 4.5 Case-size prior
- Maintain learnable expert centers in normalized log-size space.
- Convert distance-to-center into prior logits.
- Blend learned logits and prior logits by `size_prior_strength`.

## 4.6 Sparse gating
- Softmax probabilities over experts.
- Keep top-k experts per node.
- Re-normalize top-k weights.
- Weighted sum of expert outputs gives final node prediction.

## 4.7 Diagnostics and aux losses
- Routing entropy.
- Expert importance distribution (probability mass).
- Expert load distribution (selection frequency).
- CV²-based load-balance loss.
- Router z-loss.

Stored in `self.router_stats` for future integration into total objective.

---

## 5) Why this is appropriate for OOD generalization

1. **Scale-aware routing** gives explicit adaptation to case size while preserving shared representation learning.
2. **Top-k sparse mixing** allows smooth interpolation for ambiguous samples (e.g., mid-size graphs) instead of brittle hard assignment.
3. **Aux balancing** prevents expert collapse, which is critical for unseen test settings.
4. **Shared encoder + separate experts** yields a good bias-variance tradeoff:
   - shared stem captures universal physics-like motifs,
   - experts model regime-specific nonlinearities.

---

## 6) Training plan (no code changes required now)

## Phase A: Standalone smoke runs
- Start with `n_experts=4`, `top_k=2`, `size_prior_strength=0.35`.
- Use existing PF loss stack (MSE + masked + PBL).
- Confirm stable convergence and non-collapsed expert loads.

## Phase B: Ablations
- `top_k`: 1 vs 2 vs 3
- `n_experts`: 3 vs 4 vs 6
- `size_prior_strength`: 0.0 (pure learned) vs 0.2 vs 0.35 vs 0.6
- `size_feature_mode`: nodes-only vs log-n/e vs nodes-edges-density
- noisy gating: off vs small std (0.05–0.2)

## Phase C: OOD-focused evaluation
- Train on subsets of sizes and evaluate on held-out sizes.
- Report per-size errors and routing distributions.
- Inspect whether held-out sizes are routed to semantically neighboring experts.

## Phase D: Integrate MoE aux losses in objective (future)
- Add explicit loss hooks (e.g., recycle/custom loss consuming `model.router_stats`).
- Tune `load_balance_coef` and `z_loss_coef` jointly with task loss.

---

## 7) Suggested next extensions (future iterations)

1. **Heterogeneous experts**:
   - mix GraphConv, TAGConv, and/or PFNet-style expert blocks.
2. **Hierarchical routing**:
   - graph-level coarse expert selection + node-level fine routing.
3. **Weak/strong MoE variant**:
   - MLP weak expert + GNN strong experts, confidence-gated.
4. **Topology-aware routing features**:
   - average degree, spectral proxies, contingency type embedding.
5. **Distillation from teacher ensemble**:
   - pretrain experts from specialized single-size teachers.

---

## 8) Operational summary

The newly added model is intentionally self-contained and minimally invasive:
- New file only (`core/models/graph_moe.py`).
- Registry auto-discovers it via existing `load_registry()` behavior.
- Existing trainer/loss wiring remains untouched.

This gives a practical first MoE baseline for PFDelta that is explicitly aligned with your objective: **single model, case-size-aware routing, improved OOD behavior potential**.
