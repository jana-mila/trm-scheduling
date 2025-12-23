import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from omegaconf import DictConfig

# Import your model and helper
from src.TinyRecursiveModels_regression.models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from src.utils import is_solution_close

class TinyRecursiveModelJobShop(pl.LightningModule):
    def __init__(self, config: DictConfig, lr: float, act_loss_weight: float, epsilon: float):
        super().__init__()
        self.save_hyperparameters()
        self.model = TinyRecursiveReasoningModel_ACTV1(dict(config))
        self.loss_fn = nn.MSELoss(reduction='none')
        self.act_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, carry, batch):
        return self.model(carry = carry, batch = batch)

    def _shared_step(self, batch):
        """A single step of the recursive model, used by train and val"""
        carry = self.model.initial_carry(batch)
        carry = carry.to(self.device)
        total_loss_for_batch = 0
        is_solution_correct = None
        halt_max_steps = self.hparams.config.halt_max_steps

        for step in range(halt_max_steps):
            carry, output = self(carry = carry, batch = batch)
            y_hat = output['logits']
            y_true = batch['labels']
            mask = batch['mask']

            # 1. Regression loss
            step_loss_masked = self.loss_fn(y_hat, y_true) * mask
            num_real_tokens = mask.sum()
            task_loss = step_loss_masked.sum() / num_real_tokens

            # 2. ACT loss
            is_solution_correct = is_solution_close(y_hat, y_true, mask, epsilon=self.hparams.epsilon)
            halt_target = is_solution_correct.float() # Turns into binary
            act_loss = self.act_loss_fn(output['q_halt_logits'], halt_target)

            # 3. Combine
            total_step_loss = task_loss + (self.hparams.act_loss_weight * act_loss)
            total_loss_for_batch += total_step_loss

            if torch.all(carry.halted):
                break

        avg_loss = total_loss_for_batch  / halt_max_steps
        accuracy = is_solution_correct.sum() / len(batch['inputs'])

        return avg_loss, accuracy

    def training_step(self, batch, batch_idx):
        avg_loss, accuracy = self._shared_step(batch)

        # log metrics
        self.log('train_loss', avg_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', accuracy, on_step=False, on_epoch=True)

        return avg_loss

    def validation_step(self, batch, batch_idx):
        avg_loss, accuracy = self._shared_step(batch)
        
        # Log metrics
        self.log('val_loss', avg_loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_epoch=True)
        
        return avg_loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer
            

if __name__ == "__main__":
    print(f"Running {__file__} as a script for testing")