import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from omegaconf import DictConfig

from src.TinyRecursiveModels_classification.models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

class TinyRecursiveModelJobShop(pl.LightningModule):
    def __init__(self, config: DictConfig, lr: float, act_loss_weight: float):
        super().__init__()
        self.save_hyperparameters()

        # The inner model is bidirectional (causal=False), fitting JSSP perfectly.
        self.model = TinyRecursiveReasoningModel_ACTV1(dict(config))

        # reduction='none' is REQUIRED to manually apply the padding mask later
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

        # ACT halting loss
        self.act_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, carry, batch):
        return self.model(carry=carry, batch=batch)

    def _shared_step(self, batch):
        # 1. Initialize Memory (Carry)
        carry = self.model.initial_carry(batch)
        
        if hasattr(carry, 'inner_carry'):
            carry.inner_carry.z_H = carry.inner_carry.z_H.to(self.device)
            carry.inner_carry.z_L = carry.inner_carry.z_L.to(self.device)
            carry.steps = carry.steps.to(self.device)
            carry.halted = carry.halted.to(self.device)
            carry.current_data = {k: v.to(self.device) for k, v in carry.current_data.items()}

        total_loss_for_batch = 0.0
        is_solution_correct = None
        halt_max_steps = self.hparams.config.halt_max_steps

        # Targets: (B, L, K)
        targets = batch["labels"].long()
        B, L, K = targets.shape

        # Mask: (B, L, K)
        if "mask" in batch:
            mask = batch["mask"].float()
            # If mask is (B, L, 1), broadcast to (B, L, K)
            if mask.shape[-1] == 1:
                mask = mask.expand(B, L, K)
        else:
            mask = torch.ones(B, L, K, device=self.device, dtype=torch.float)

        # -----------------------------------------------------------------------
        # DEBUG: Check for out-of-bounds labels immediately (Run once then remove)
        # -----------------------------------------------------------------------
        # num_classes = 10 (Hardcoded in your inner model)
        # max_label = targets.max().item()
        # if max_label >= 10:
        #     raise ValueError(f"CRITICAL ERROR: Found label {max_label}, but model only outputs 10 classes (0-9).")
        # -----------------------------------------------------------------------

        for step in range(halt_max_steps):
            carry, output = self(carry=carry, batch=batch)
            logits = output["logits"]
            num_classes = logits.shape[-1]

            # Flatten
            flat_logits  = logits.reshape(B * L * K, num_classes)
            flat_targets = targets.reshape(B * L * K)
            flat_mask    = mask.reshape(B * L * K)
            
            # --- CRITICAL FIX START ---
            # We must ensure that wherever the mask is 0 (padding), the target is -100.
            # Otherwise, if padding is -1 or 0, it might crash the GPU or confuse the model.
            
            # Clone targets so we don't modify the original batch in place
            safe_targets = flat_targets.clone()
            
            # Set padded positions to -100
            safe_targets[flat_mask == 0] = -100
            # --- CRITICAL FIX END ---

            # 1. Classification Loss
            # Now safe_targets contains only valid labels (0-9) or -100.
            raw_loss = self.ce_loss(flat_logits, safe_targets) 
            
            # Apply Mask (Redundant if using -100, but good for safety)
            masked_loss = raw_loss * flat_mask
            num_valid = flat_mask.sum() + 1e-8
            task_loss = masked_loss.sum() / num_valid

            # 2. Accuracy
            preds = logits.argmax(dim=-1)
            matches = (preds == targets)
            
            # Correct if: Matches OR Target is -100 OR Padding
            ignore_mask = (targets == -100)
            is_padding  = (mask == 0)
            effective_correctness = matches | ignore_mask | is_padding
            
            is_sample_correct = effective_correctness.all(dim=-1).all(dim=-1)

            # 3. ACT Loss
            halt_target = is_sample_correct.float()
            q_halt = output["q_halt_logits"].squeeze(-1)
            if q_halt.ndim > 1: q_halt = q_halt.view(B)

            act_loss_raw = self.act_loss_fn(q_halt, halt_target)
            
            # Mask ACT loss based on sample validity
            sample_valid_mask = mask.reshape(B, -1).max(dim=1).values
            act_loss = (act_loss_raw * sample_valid_mask).sum() / (sample_valid_mask.sum() + 1e-8)

            total_step_loss = task_loss + self.hparams.act_loss_weight * act_loss
            total_loss_for_batch += total_step_loss

            if torch.all(carry.halted):
                break

        avg_loss = total_loss_for_batch / halt_max_steps
        
        if sample_valid_mask.sum() > 0:
            accuracy = is_sample_correct[sample_valid_mask.bool()].float().mean()
        else:
            accuracy = torch.tensor(0.0, device=self.device)

        return avg_loss, accuracy

    def training_step(self, batch, batch_idx):
        avg_loss, accuracy = self._shared_step(batch)
        self.log("train_loss", avg_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", accuracy, on_epoch=True, prog_bar=True)
        return avg_loss

    def validation_step(self, batch, batch_idx):
        avg_loss, accuracy = self._shared_step(batch)
        self.log("val_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", accuracy, on_epoch=True, prog_bar=True)
        return avg_loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.hparams.lr)


if __name__ == "__main__":
    print(f"Running {__file__} as a script for testing")