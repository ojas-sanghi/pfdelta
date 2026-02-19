"""Graph Mixture-of-Experts model for PFDelta.

This module implements a node-level sparse MoE architecture that is designed to
integrate cleanly with the project's existing model/registry conventions.

Design goals:
1) Keep the external interface consistent with existing PF models: forward(data)
   returns node predictions for bus-level targets.
2) Support explicit case-size specialization (the main requirement for OOD
   generalization across grid sizes) while also allowing learned adjustments.
3) Expose diagnostics (routing entropy, expert loads, aux losses) without
   changing the trainer API.

The model combines:
- A shared encoder over bus input features.
- Multiple GraphConv expert towers (same output dimensionality).
- A router that uses node embeddings + graph-level size statistics.
- A sparse top-k gating policy with optional noise during training.
- Optional load-balancing and z-loss terms stored for downstream use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphConv

from core.utils.registry import registry


@dataclass
class RouterStats:
    """Container for MoE routing diagnostics from the latest forward pass."""

    routing_probs: Tensor
    routing_weights: Tensor
    topk_indices: Tensor
    entropy: Tensor
    mean_entropy: Tensor
    expert_importance: Tensor
    expert_load: Tensor
    load_balance_loss: Tensor
    z_loss: Tensor


class GraphConvExpert(nn.Module):
    """A GraphConv expert tower used inside GraphCaseSizeMoE.

    This follows the style of existing graph_conv model in the repository but
    isolates expert behavior so multiple independent towers can be instantiated.
    """

    def __init__(self, hidden_dim: int, out_dim: int, n_layers: int):
        super().__init__()
        if n_layers < 2:
            raise ValueError("n_layers for each expert must be >= 2")

        self.start_conv = GraphConv(hidden_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [GraphConv(hidden_dim, hidden_dim) for _ in range(n_layers - 2)]
        )
        self.end_conv = GraphConv(hidden_dim, out_dim)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.start_conv(x=x, edge_index=edge_index))
        for layer in self.layers:
            x = F.relu(layer(x=x, edge_index=edge_index))
        x = self.end_conv(x=x, edge_index=edge_index)
        return x


@registry.register_model("graph_moe")
class GraphCaseSizeMoE(nn.Module):
    """Case-size-aware sparse MoE model for PFDelta bus-level predictions.

    Parameters
    ----------
    in_dim : int
        Input node feature dimensionality after slicing bus.x.
    hidden_dim : int
        Shared latent dimension before experts.
    out_dim : int
        Output node feature dimension (e.g., 6 for PF targets).
    n_layers : int
        Number of GraphConv layers per expert tower (>= 2).
    n_experts : int, default=4
        Number of experts in the MoE layer.
    top_k : int, default=2
        Number of experts selected per node during sparse routing.
    router_hidden_dim : int, default=128
        Hidden dimension for the router MLP.
    size_feature_mode : str, default="log_nodes_edges"
        Which graph-size signal to provide to router: one of
        {"nodes_only", "log_nodes_edges", "nodes_edges_density"}.
    size_temperature : float, default=1.0
        Temperature controlling the sharpness of deterministic size prior.
    size_prior_strength : float, default=0.35
        Weight in [0, 1] controlling blend between learned logits and
        deterministic size-bucket prior logits.
    noisy_gating_std : float, default=0.0
        Gaussian noise standard deviation added to router logits during training.
    use_router_layernorm : bool, default=True
        Whether to apply LayerNorm in the router MLP.
    load_balance_coef : float, default=0.01
        Coefficient applied to load-balance auxiliary loss (stored only).
    z_loss_coef : float, default=0.001
        Coefficient applied to router z-loss (stored only).
    x_start : int, default=4
        Start column of bus.x slice (inclusive), matching existing PF models.
    x_end : int, default=10
        End column of bus.x slice (exclusive), matching existing PF models.

    Notes
    -----
    - The output remains a tensor so existing training/loss pipelines work.
    - Auxiliary MoE losses are *not* automatically added to task loss, because
      that would require changing the loss stack. They are exposed in
      ``self.router_stats`` for optional integration.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
        n_experts: int = 4,
        top_k: int = 2,
        router_hidden_dim: int = 128,
        size_feature_mode: str = "log_nodes_edges",
        size_temperature: float = 1.0,
        size_prior_strength: float = 0.35,
        noisy_gating_std: float = 0.0,
        use_router_layernorm: bool = True,
        load_balance_coef: float = 0.01,
        z_loss_coef: float = 0.001,
        x_start: int = 4,
        x_end: int = 10,
    ):
        super().__init__()

        if n_experts < 2:
            raise ValueError("n_experts must be >= 2 for MoE")
        if top_k < 1 or top_k > n_experts:
            raise ValueError("top_k must satisfy 1 <= top_k <= n_experts")
        if size_feature_mode not in {
            "nodes_only",
            "log_nodes_edges",
            "nodes_edges_density",
        }:
            raise ValueError(
                "size_feature_mode must be one of "
                "{'nodes_only','log_nodes_edges','nodes_edges_density'}"
            )

        self.n_experts = n_experts
        self.top_k = top_k
        self.noisy_gating_std = noisy_gating_std
        self.load_balance_coef = load_balance_coef
        self.z_loss_coef = z_loss_coef
        self.size_feature_mode = size_feature_mode
        self.size_temperature = size_temperature
        self.size_prior_strength = size_prior_strength
        self.x_start = x_start
        self.x_end = x_end

        self.input_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        size_feat_dim = {
            "nodes_only": 1,
            "log_nodes_edges": 2,
            "nodes_edges_density": 3,
        }[size_feature_mode]

        router_layers: List[nn.Module] = [
            nn.Linear(hidden_dim + size_feat_dim, router_hidden_dim)
        ]
        if use_router_layernorm:
            router_layers.append(nn.LayerNorm(router_hidden_dim))
        router_layers.extend([nn.ReLU(), nn.Linear(router_hidden_dim, n_experts)])
        self.router = nn.Sequential(*router_layers)

        self.experts = nn.ModuleList(
            [GraphConvExpert(hidden_dim, out_dim, n_layers) for _ in range(n_experts)]
        )

        # Learnable centers in normalized log-node-size space for size prior.
        default_centers = torch.linspace(0.0, 1.0, steps=n_experts)
        self.size_centers = nn.Parameter(default_centers)
        self.router_stats: Optional[RouterStats] = None

    def _extract_graph_batch(self, data: HeteroData, num_nodes: int) -> Tensor:
        """Return graph IDs per node, supporting single-graph and mini-batches."""
        if hasattr(data["bus"], "batch") and data["bus"].batch is not None:
            return data["bus"].batch
        return torch.zeros(num_nodes, dtype=torch.long, device=data["bus"].x.device)

    def _graph_size_features(
        self,
        batch: Tensor,
        edge_index: Tensor,
        num_graphs: int,
    ) -> Tuple[Tensor, Tensor]:
        """Compute per-graph and per-node size features for routing.

        Returns
        -------
        node_size_features : Tensor
            Shape [num_nodes, size_feat_dim].
        node_log_sizes_norm : Tensor
            Shape [num_nodes], normalized log(|V|) used by size prior.
        """
        device = batch.device
        num_nodes_per_graph = torch.bincount(batch, minlength=num_graphs).float()

        edge_src_graph = batch[edge_index[0]]
        num_edges_per_graph = torch.bincount(edge_src_graph, minlength=num_graphs).float()

        n = num_nodes_per_graph.clamp_min(1.0)
        e = num_edges_per_graph.clamp_min(0.0)
        log_n = torch.log(n)
        log_e = torch.log1p(e)
        density = e / (n * (n - 1.0)).clamp_min(1.0)

        log_n_norm = (log_n - log_n.min()) / (log_n.max() - log_n.min() + 1e-6)

        if self.size_feature_mode == "nodes_only":
            graph_feats = n.unsqueeze(-1)
        elif self.size_feature_mode == "log_nodes_edges":
            graph_feats = torch.stack([log_n, log_e], dim=-1)
        else:
            graph_feats = torch.stack([n, e, density], dim=-1)

        graph_feats = (graph_feats - graph_feats.mean(0, keepdim=True)) / (
            graph_feats.std(0, keepdim=True, unbiased=False) + 1e-6
        )

        node_feats = graph_feats[batch]
        node_log_n_norm = log_n_norm[batch]
        return node_feats, node_log_n_norm

    def _size_prior_logits(self, node_log_n_norm: Tensor) -> Tensor:
        """Deterministic case-size prior logits for experts.

        Each expert has a learnable center in normalized log-size space.
        We score proximity to each center and scale by size_temperature.
        """
        dist2 = (node_log_n_norm.unsqueeze(-1) - self.size_centers.unsqueeze(0)) ** 2
        logits = -dist2 / max(self.size_temperature, 1e-6)
        return logits

    def _sparse_topk_gating(self, logits: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute sparse top-k gate values per node.

        Returns
        -------
        probs : Tensor
            Dense softmax probabilities over experts, shape [N, E].
        sparse_weights : Tensor
            Sparse normalized top-k weights with zeros elsewhere, shape [N, E].
        topk_idx : Tensor
            Top-k expert indices per node, shape [N, K].
        """
        if self.training and self.noisy_gating_std > 0.0:
            logits = logits + torch.randn_like(logits) * self.noisy_gating_std

        probs = F.softmax(logits, dim=-1)
        topk_vals, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)
        topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-8)

        sparse_weights = torch.zeros_like(probs)
        sparse_weights.scatter_(dim=-1, index=topk_idx, src=topk_vals)
        return probs, sparse_weights, topk_idx

    def _compute_aux_losses(
        self,
        probs: Tensor,
        sparse_weights: Tensor,
        logits: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Compute routing diagnostics and optional auxiliary losses."""
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        mean_entropy = entropy.mean()

        # Importance: how much gate mass each expert receives.
        importance = probs.sum(dim=0)
        importance = importance / (importance.sum() + 1e-8)

        # Load: how often each expert is selected by top-k sparse routing.
        load = (sparse_weights > 0).float().sum(dim=0)
        load = load / (load.sum() + 1e-8)

        # Encourage balanced expert usage; CV^2 style.
        imp_cv2 = torch.var(importance) / (torch.mean(importance) ** 2 + 1e-8)
        load_cv2 = torch.var(load) / (torch.mean(load) ** 2 + 1e-8)
        load_balance_loss = self.load_balance_coef * (imp_cv2 + load_cv2)

        # Router z-loss from large-scale MoE literature for logit regularization.
        z = torch.logsumexp(logits, dim=-1)
        z_loss = self.z_loss_coef * torch.mean(z**2)

        return entropy, mean_entropy, importance, load, load_balance_loss, z_loss

    def forward(self, data: HeteroData) -> Tensor:
        """Forward pass returning bus-level predictions."""
        x = data["bus"].x[:, self.x_start : self.x_end]
        edge_index = data["bus", "branch", "bus"].edge_index

        h = self.input_encoder(x)

        batch = self._extract_graph_batch(data, num_nodes=h.size(0))
        num_graphs = int(batch.max().item()) + 1

        node_size_features, node_log_n_norm = self._graph_size_features(
            batch=batch,
            edge_index=edge_index,
            num_graphs=num_graphs,
        )

        router_inputs = torch.cat([h, node_size_features], dim=-1)
        learned_logits = self.router(router_inputs)
        prior_logits = self._size_prior_logits(node_log_n_norm)
        router_logits = (
            (1.0 - self.size_prior_strength) * learned_logits
            + self.size_prior_strength * prior_logits
        )

        probs, sparse_weights, topk_idx = self._sparse_topk_gating(router_logits)

        expert_outputs = [expert(h, edge_index) for expert in self.experts]
        stacked = torch.stack(expert_outputs, dim=1)  # [N, E, out_dim]
        out = torch.sum(stacked * sparse_weights.unsqueeze(-1), dim=1)

        entropy, mean_entropy, importance, load, load_balance_loss, z_loss = self._compute_aux_losses(
            probs=probs,
            sparse_weights=sparse_weights,
            logits=router_logits,
        )
        self.router_stats = RouterStats(
            routing_probs=probs.detach(),
            routing_weights=sparse_weights.detach(),
            topk_indices=topk_idx.detach(),
            entropy=entropy.detach(),
            mean_entropy=mean_entropy.detach(),
            expert_importance=importance.detach(),
            expert_load=load.detach(),
            load_balance_loss=load_balance_loss.detach(),
            z_loss=z_loss.detach(),
        )

        return out
