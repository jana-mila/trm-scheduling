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

#%%
# -----------------------------
# CONFIG
# -----------------------------
# --- Hyperparameters ---
PROBLEMS_DIR = 'data/problems'
SOLUTIONS_DIR = 'data/solutions'
BATCH_SIZE = 4
NUM_EPOCHS = 10
LR = 1e-4
HALT_MAX_STEPS = 16 # This is N_sup, the number of "Deep Supervision" steps
ACT_LOSS_WEIGHT = 0.01  # Weight for the ACT (halting) loss

# --- Model & Data Shape Config ---
# We must pad all problems to a fixed size.
MAX_J = 20
MAX_T = 20
MAX_SEQ_LEN = MAX_J * MAX_T # This is the 'seq_len' for the model
HIDDEN_SIZE = 256 # Model's internal dimension
PUZZLE_EMB_DIM = 64 # Dimension for the "task ID" embedding

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
# TRAINING LOOP
# -----------------------------
print("Starting training...")
dataset = CustomJobShopDataset(PROBLEMS_DIR, SOLUTIONS_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)

# %%
for epoch in range(NUM_EPOCHS):
    print(f"\n ---Epoch {epoch+1}/{NUM_EPOCHS} ---")
    model.train()

    for batch in tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        carry = model.initial_carry(batch)
        total_loss_for_batch = 0
        for step in range(HALT_MAX_STEPS):
            carry, outputs = model(carry=carry, batch=batch)

            # This is the *full* output: (B, 416, 1)
            logits = outputs['logits'] 
            # full_logits = outputs['logits'] 
            
            # # This is the *sliced* output: (B, 400, 1)
            # logits = full_logits[:, model.config.puzzle_emb_len:] 
            labels = batch['labels']
            mask = batch['mask'] # (B, L, 1)

            # --- 1. Calculate Main Task Loss (Regression) ---
            step_loss_unmasked = loss_fn(logits, labels)
            step_loss_masked = step_loss_unmasked * mask
            
            # Add a small epsilon to prevent division by zero if a batch has no real tokens
            # (which shouldn't happen, but is good practice)
            num_real_tokens = mask.sum() + 1e-8 
            task_loss = step_loss_masked.sum() / num_real_tokens

            # --- 2. Calculate ACT Loss (Halting) ---
            # We want the model to halt when its answer is "good enough"
            # For regression, "good enough" is complex. A simpler target
            # is to just train it to halt on the *last step*.
            
            # Target is 0 (don't halt) for all steps except the last
            is_last_step = (step == HALT_MAX_STEPS - 1)
            halt_target = torch.full_like(outputs['q_halt_logits'], 
                                          float(is_last_step))
            
            act_loss = act_loss_fn(outputs['q_halt_logits'], halt_target)

            # --- 3. Combine Losses ---
            total_step_loss = task_loss + (act_loss * ACT_LOSS_WEIGHT)

            total_loss_for_batch += total_step_loss
        
         # Backpropagate the *sum* of losses from all 16 steps
        optimizer.zero_grad()
        total_loss_for_batch.backward()
        optimizer.step()    
    print(f"Epoch {epoch+1} finished. Final batch avg loss: {total_loss_for_batch.item() / halt_max_steps:.6f}")


