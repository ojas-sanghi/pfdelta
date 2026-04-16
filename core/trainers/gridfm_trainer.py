"""
GridFM Trainer for PFDelta.

Extends GNNTrainer with GridFM-specific features:
  - Automatic injection of the model reference into GridFM losses.
  - Support for zero-shot evaluation (epochs=0).
  - Support for fine-tuning from a pre-trained GridFM checkpoint.
  - Freeze / unfreeze of encoder layers for staged fine-tuning.

YAML config example (zero-shot)::

    functional:
      trainer_name: gridfm_trainer

YAML config example (fine-tuning with partial freeze)::

    functional:
      trainer_name: gridfm_trainer
      freeze_encoder: true      # freeze encoder + first N GPS layers
      freeze_gps_layers: 4      # freeze first 4 GPS layers
"""

import copy
import torch
from tqdm import tqdm
from torch_geometric.loader.dataloader import DataLoader

from core.trainers.gnn_trainer import GNNTrainer
from core.utils.registry import registry


@registry.register_trainer("gridfm_trainer")
class GridFMTrainer(GNNTrainer):
    """Trainer specialised for the GridFMWrapper model.

    Inherits the full training loop from GNNTrainer and adds:

    1.  **Model injection into losses** – GridFM losses (gridfm_masked_mse,
        gridfm_pbe, etc.) need a reference to the model for normalisation.
        After losses are loaded, this trainer calls ``loss.set_model(model)``
        on any loss that exposes that method.

    2.  **Partial parameter freezing** – controlled by:
        ``functional.freeze_encoder`` (bool) and
        ``functional.freeze_gps_layers`` (int, number of GPS layers to freeze).

    3.  **Zero-shot mode** – if ``train_params.epochs == 0`` the trainer skips
        training entirely and only runs validation.
    """

    def setup_training(self):
        super().setup_training()
        # After model and losses are loaded, inject model into GridFM losses
        self._inject_model_into_losses()
        # Apply freezing if requested
        self._apply_freezing()

    # ------------------------------------------------------------------
    # Loss injection
    # ------------------------------------------------------------------

    def _inject_model_into_losses(self):
        """Pass the model reference to any GridFM loss that needs it."""
        for loss in self.train_loss + self.val_loss:
            if hasattr(loss, "set_model"):
                loss.set_model(self.model)

    # ------------------------------------------------------------------
    # Partial freezing
    # ------------------------------------------------------------------

    def _apply_freezing(self):
        functional = self.config.get("functional", {})
        freeze_encoder = functional.get("freeze_encoder", False)
        freeze_gps_layers = int(functional.get("freeze_gps_layers", 0))

        if not freeze_encoder and freeze_gps_layers == 0:
            return

        # Access GPSTransformer (GridFMWrapper stores it as .gps)
        gps = getattr(self.model, "gps", self.model)

        if freeze_encoder and hasattr(gps, "encoder"):
            for p in gps.encoder.parameters():
                p.requires_grad_(False)
            if hasattr(gps, "input_norm"):
                for p in gps.input_norm.parameters():
                    p.requires_grad_(False)
            if hasattr(gps, "pe_norm"):
                for p in gps.pe_norm.parameters():
                    p.requires_grad_(False)
            print(f"[GridFMTrainer] Encoder frozen.")

        if freeze_gps_layers > 0 and hasattr(gps, "layers"):
            for i in range(min(freeze_gps_layers, len(gps.layers))):
                for p in gps.layers[i].parameters():
                    p.requires_grad_(False)
            print(f"[GridFMTrainer] Froze first {freeze_gps_layers} GPS layers.")

        # Print trainable parameter count after freezing
        n_trainable = sum(p.numel() for p in self.model.parameters()
                          if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        print(f"[GridFMTrainer] Trainable params: {n_trainable:,} / {n_total:,}")

    # ------------------------------------------------------------------
    # Override train() to support zero-shot (epochs=0)
    # ------------------------------------------------------------------

    def train(self):
        train_params = self.config["optim"]["train_params"]
        epochs = train_params.get("epochs", 0)
        if epochs == 0:
            print("\n[GridFMTrainer] epochs=0 → zero-shot evaluation only.\n")
            self._run_zero_shot_eval()
            return
        # Normal training
        super().train()

    def _run_zero_shot_eval(self):
        """Evaluate all validation sets with no training."""
        self.model.eval()
        for val_idx, val_dl in enumerate(self.dataloaders[1:]):
            running = [0.0] * len(self.val_loss)
            for data in tqdm(val_dl, desc=f"Zero-shot val set {val_idx+1}"):
                data = data.to(self.device)
                outputs = self.model(data)
                for i, loss_fn in enumerate(self.val_loss):
                    running[i] += loss_fn(outputs, data).item()
            n = len(val_dl)
            if n > 0:
                for i, name in enumerate(self.val_loss_names):
                    avg = running[i] / n
                    print(f"  Val set {val_idx+1} | {name}: {avg:.6f}")

    # ------------------------------------------------------------------
    # Override get_dataloader_class to use PyG's DataLoader
    # ------------------------------------------------------------------

    def get_dataloader_class(self):
        return DataLoader
