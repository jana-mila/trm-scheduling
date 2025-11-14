#%%
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / "TinyRecursiveModels"))
from TinyRecursiveModels.models.recursive_reasoning.trm import *

import torch 
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Optional
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path

import torch.optim as optim
from tqdm import tqdm
import wandb
from datetime import datetime

#%%
# -----------------------------
# CONFIG
# -----------------------------
# --- Hyperparameters ---
TRAIN_PROBLEMS_DIR = 'data/train/problems'
TRAIN_SOLUTIONS_DIR = 'data/train/solutions'
VAL_PROBLEMS_DIR = 'data/val/problems'
VAL_SOLUTIONS_DIR = 'data/val/solutions'
CHKPT_DIR = 'checkpoints'
CHKPT_DIR = Path(CHKPT_DIR)

BATCH_SIZE = 4
NUM_EPOCHS = 10
LR = 1e-4
HALT_MAX_STEPS = 128 # This is N_sup, the number of "Deep Supervision" steps
ACT_LOSS_WEIGHT = 0.01  # Weight for the ACT (halting) loss

# --- Model & Data Shape Config ---
# We must pad all problems to a fixed size.
MAX_J = 20
MAX_T = 20
MAX_SEQ_LEN = MAX_J * MAX_T # This is the 'seq_len' for the model
HIDDEN_SIZE = 256 # Model's internal dimension
PUZZLE_EMB_DIM = 64 # Dimension for the "task ID" embedding
EPSILON = 1e-3 # How close the predicted solution to the actual solution to be considered correct

model_cfg = dict(
    batch_size=BATCH_SIZE,
    seq_len=MAX_SEQ_LEN,
    puzzle_emb_ndim=PUZZLE_EMB_DIM,
    num_puzzle_identifiers=1, # We only have one task: "JSP"
    vocab_size=0,             # We are not using a token vocab (CRITICAL)
    puzzle_emb_len=16,
    
    # Recursive reasoning config
    H_cycles=3,               # T=3 in the paper, a good default
    L_cycles=6,               # n=6 in the paper, a good default
    L_layers=2,               # "tiny" 2-layer model from paper
    H_layers=0,               # This is ignored since the TRM simplified the hierarchy

    # Transformer config
    hidden_size=HIDDEN_SIZE,
    expansion=2.0,            # Standard for SwiGLU
    num_heads=4,              # Must be a divisor of HIDDEN_SIZE
    pos_encodings="rope",

    # Halting Q-learning config
    halt_max_steps=HALT_MAX_STEPS,
    halt_exploration_prob=0.0, # Not needed for simple training
    no_ACT_continue=True,      # Use the simplified ACT loss from paper

    forward_dtype="float32"
)

#%%
# -----------------------------
# DATASET AND DATALOADER
# -----------------------------
class CustomJobShopDataset(Dataset):
    def __init__(self, problems_dir, solutions_dir, transform=None):
        self.problems_dir = Path(problems_dir)
        self.solutions_dir = Path(solutions_dir)
        self.transform = transform

        self.problems_files = sorted(self.problems_dir.glob('*.pt'))
        self.solutions_files = sorted(self.solutions_dir.glob('*.pt'))

        self.num_problems = len(self.problems_files)
        assert self.num_problems == len(self.solutions_files), "Problem and solution file count mismatch"

    def __len__(self):
        return self.num_problems

    def __getitem__(self, idx):
        problem = torch.load(self.problems_files[idx])
        solution = torch.load(self.solutions_files[idx])
        
        # flatten from (J,T,C) to (L,C)
        problem = problem.view(-1, 2)
        solution = solution.view(-1, 1)
        # we need to provide puzzle_identifier, 0 is the ID for our JSP task
        puzzle_identifier = torch.tensor(0, dtype=torch.int32)
        
        return {
            "inputs": problem,
            "labels": solution,
            "puzzle_identifiers": puzzle_identifier
        }


def custom_collate_fn(batch):
    max_len = max(item['inputs'].shape[0] for item in batch)
    if max_len > MAX_SEQ_LEN:
        max_len = MAX_SEQ_LEN

    # Pad inputs (the (M, D) puzzle)
    padded_problems = torch.zeros(len(batch), MAX_SEQ_LEN, 2, dtype=torch.float32)
    # Pad labels (the (S) solution)
    padded_solutions = torch.full((len(batch), MAX_SEQ_LEN, 1), -1.0, dtype=torch.float32)
    # Mask to identify real data vs. padding
    mask = torch.zeros(len(batch), MAX_SEQ_LEN, 1, dtype=torch.bool)
    
    identifiers = []

    for i, item in enumerate(batch):
        seq_len = item['inputs'].shape[0]
        
        # Truncate if longer than max len
        if seq_len > MAX_SEQ_LEN:
            seq_len = MAX_SEQ_LEN
            
        padded_problems[i, :seq_len] = item['inputs'][:seq_len]
        padded_solutions[i, :seq_len] = item['labels'][:seq_len]
        mask[i, :seq_len] = True
        identifiers.append(item['puzzle_identifiers'])

    return {
        'inputs': padded_problems,
        'labels': padded_solutions,
        'mask': mask,
        'puzzle_identifiers': torch.stack(identifiers)
    }

#%%
def is_solution_close(y_hat: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor, epsilon: float) -> torch.Tensor:
    abs_diff = torch.abs(y_hat - y_true)
    is_close = (abs_diff < epsilon)
    is_good = (is_close | ~mask)
    # reduces shape from [B, L, 1] -> [B]
    return is_good.all(dim=(1,2))

# %%
# -----------------------------
# INITIALIZE DATA LOADERS
# -----------------------------
print("Starting training...")
train_dataset = CustomJobShopDataset(TRAIN_PROBLEMS_DIR, TRAIN_SOLUTIONS_DIR)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
num_train_problems = len(train_dataset)
print(f"There are {num_train_problems} training job shop problem samples")

val_dataset = CustomJobShopDataset(VAL_PROBLEMS_DIR, VAL_SOLUTIONS_DIR)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
num_val_problems = len(val_dataset)
print(f"There are {num_val_problems} validation job shop problem samples")

# %%
# -----------------------------
# MODEL, LOSS, OPTIMIZER
# -----------------------------
print("Setting up model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = TinyRecursiveReasoningModel_ACTV1(model_cfg)
loss_fn = nn.MSELoss(reduction='none') # 'none' to apply mask manually
act_loss_fn = nn.BCEWithLogitsLoss() # For the halting signal
optimizer = optim.AdamW(model.parameters(), lr=LR)
print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters.")
print(f"Model is as follows: \n{model}")

# %%
# -----------------------------
# WANDB
# -----------------------------
now = datetime.now().strftime("%m%d_%H%M")
exp_name = f"{now}"
wandb.login()
wandb.init(
    project="trm-scheduling",
    name=exp_name,
    config={
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": LR,
        **model_cfg
        }
    )
wandb.watch(model, log="all", log_freq=100)

# %%
# -----------------------------
# TRAINING
# -----------------------------
best_val_loss = 1e10
for epoch in range(NUM_EPOCHS):
    print(f"\n ---Epoch {epoch+1}/{NUM_EPOCHS} ---")
    model.train()
    train_epoch_loss = 0.0
    train_total_correct = 0
    train_batch = 0

    for idx, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        carry = model.initial_carry(batch)
        total_loss_for_batch = 0

        for step in range(HALT_MAX_STEPS):
            carry, outputs = model(carry=carry, batch=batch)
            y_hat = outputs['logits']             
            y_true = batch['labels']
            mask = batch['mask'] # (B, L, 1)

            # --- 1. Calculate Main Task Loss (Regression) ---
            step_loss_unmasked = loss_fn(y_hat, y_true)
            step_loss_masked = step_loss_unmasked * mask
            
            # Add a small epsilon to prevent division by zero if a batch has no real tokens
            # (which shouldn't happen, but is good practice)
            num_real_tokens = mask.sum() + 1e-8 
            task_loss = step_loss_masked.sum() / num_real_tokens

            # --- 2. Calculate ACT Loss (Halting) ---
            # We want the model to halt when its answer is "good enough"
            # For regression, "good enough" is complex, we use absolute difference.
            is_solution_correct = is_solution_close(y_hat, y_true, mask, epsilon=EPSILON)
            halt_target = is_solution_correct.float()
            act_loss = act_loss_fn(outputs['q_halt_logits'], halt_target)

            # --- 3. Combine Losses ---
            total_step_loss = task_loss + (act_loss * ACT_LOSS_WEIGHT)
            total_loss_for_batch += total_step_loss
        
        # Backpropagate the *sum* of losses from all 16 steps
        optimizer.zero_grad()
        total_loss_for_batch.backward()
        optimizer.step()

        train_batch += 1  
        train_total_correct += is_solution_correct.sum().item()
        train_epoch_loss += total_loss_for_batch.item()


    avg_train_epoch_loss = (train_epoch_loss / train_batch) / HALT_MAX_STEPS
    train_accuracy = train_total_correct / num_train_problems
    print(f"Epoch {epoch+1} finished. Average training loss: {avg_train_epoch_loss:.6f}")
    print(f"Epoch {epoch+1} finished. Total number of correct solutions: {train_total_correct}/{num_train_problems}")

    # -----------------------------
    # VALIDATE
    # -----------------------------
    model.eval()
    val_epoch_total_loss = 0.0
    val_total_correct = 0
    val_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, leave=False, desc="Validating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            carry = model.initial_carry(batch)
            total_val_loss_for_batch = 0

            for step in range(HALT_MAX_STEPS):
                carry, outputs = model(carry=carry, batch=batch)
                y_hat = outputs['logits']
                y_true = batch['labels']
                mask = batch['mask']

                step_loss_unmasked = loss_fn(y_hat, y_true)
                step_loss_masked = step_loss_unmasked * mask

                num_real_tokens = mask.sum() + 1e-8 
                task_loss = step_loss_masked.sum() / num_real_tokens

                is_solution_correct = is_solution_close(y_hat, y_true, mask, epsilon=EPSILON)
                halt_target = is_solution_correct.float()
                act_loss = act_loss_fn(outputs['q_halt_logits'], halt_target)

                total_step_loss = task_loss + (act_loss * ACT_LOSS_WEIGHT)
                total_loss_for_batch += total_step_loss
            
            val_total_correct += is_solution_correct.sum().item()
            val_epoch_total_loss += total_loss_for_batch.item()
            val_batches += 1

    val_avg_loss = (val_epoch_total_loss / val_batches) / HALT_MAX_STEPS
    val_accuracy = val_total_correct / num_val_problems
    print(f"Epoch {epoch+1} Validation loss: {val_avg_loss: .6f}") 
    print(f"Epoch {epoch+1} Validation total correct solutions: {val_total_correct} /  {num_val_problems}")
    
    wandb.log({
        'train_loss_epoch': avg_train_epoch_loss,
        'val_loss_epoch': val_avg_loss,
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy
    }, step=epoch)

    # -----------------------------
    # CHECKPOINT (IF BEST)
    # -----------------------------
    if val_avg_loss < best_val_loss:
        chckpt_name = f"{exp_name}-epoch-{epoch}-valloss-{val_avg_loss:.4f}.pth"
        chkpt_file = CHKPT_DIR / chckpt_name
        torch.save(model.state_dict(), chkpt_file)
        best_val_loss = val_avg_loss
        artifact = wandb.Artifact(name=chckpt_name, type='model')
        artifact.add_file(chkpt_file)
        wandb.log_artifact(artifact)
        print(f"New best model saved at {chkpt_file}")


wandb.finish()
print('Training complete!')


# %%
