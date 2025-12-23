import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pytorch_lightning as pl
from omegaconf import DictConfig # Import for type hinting the config

class CustomJobShopDataset(Dataset):
    # Added input_feature_dim to __init__
    def __init__(self, problems_dir, solutions_dir, input_feature_dim):
        self.problems_dir = Path(problems_dir)
        self.solutions_dir = Path(solutions_dir)
        self.input_feature_dim = input_feature_dim # Store the dimension

        self.problems_files = sorted(self.problems_dir.glob('*.pt'))
        self.solutions_files = sorted(self.solutions_dir.glob('*.pt'))

        print(f"There are {len(self.problems_files)} problems")
        print(f"There are {len(self.solutions_files)} solutions")
        self.num_problems = len(self.problems_files)
        assert self.num_problems == len(self.solutions_files), "Problem and solution file count mismatch"

    def __len__(self):
        return self.num_problems

    def __getitem__(self, idx):
        problem = torch.load(self.problems_files[idx])
        solution = torch.load(self.solutions_files[idx])
        
        input_feature_dim = problem.shape[-1]
        output_feature_dim = solution.shape[-1]
        
        # Use the variable input_feature_dim for reshaping
        problem = problem.view(-1, input_feature_dim)
        solution = solution.view(-1, output_feature_dim) # Solution/Label dimension (StartTime) is fixed at 1

        # # Use the variable input_feature_dim for reshaping
        # problem = problem.view(-1, self.input_feature_dim)
        # solution = solution.view(-1, ) # Solution/Label dimension (StartTime) is fixed at 1
        
        puzzle_identifier = torch.tensor(0, dtype=torch.int32)
        
        return {
            "inputs": problem,
            "labels": solution,
            "puzzle_identifiers": puzzle_identifier
        }

def custom_collate_fn(batch, max_seq_len, input_feature_dim: int):
    # Use the variable input_feature_dim for padding tensor initialization
    padded_problems = torch.zeros(len(batch), max_seq_len, input_feature_dim, dtype=torch.float32)
    
    # Label dimension is fixed at 1
    padded_solutions = torch.full((len(batch), max_seq_len, 1), -1.0, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_seq_len, 1, dtype=torch.bool)
    
    identifiers = []

    for i, item in enumerate(batch):
        seq_len = item['inputs'].shape[0]
        if seq_len > max_seq_len:
            seq_len = max_seq_len
            
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

# --- 3. JobShopDataModule (Propagates input_feature_dim) ---

class JobShopDataModule(pl.LightningDataModule):
    # Get config as DictConfig or dict, and pass input_feature_dim separately
    def __init__(self, config: DictConfig, batch_size: int, max_seq_len: int):
        super().__init__()
        self.cfg = config.data
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.input_feature_dim = config.model.trm_model.input_feature_dim # Dirty fix for now
        
        # Update collate_fn to pass the required dimension
        self.collate_fn = lambda x: custom_collate_fn(x, self.max_seq_len, self.input_feature_dim)

    def setup(self, stage=None):
        # Pass input_feature_dim to the Dataset constructors
        self.train_dataset = CustomJobShopDataset(
            self.cfg.train_problems_dir, 
            self.cfg.train_solutions_dir,
            input_feature_dim=self.input_feature_dim
        )
        self.val_dataset = CustomJobShopDataset(
            self.cfg.val_problems_dir, 
            self.cfg.val_solutions_dir,
            input_feature_dim=self.input_feature_dim
        )
        print(f"Train samples: {len(self.train_dataset)}, Val samples: {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            collate_fn = self.collate_fn,
            drop_last = True,
            num_workers = self.cfg.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            collate_fn = self.collate_fn,
            drop_last = True,
            num_workers = self.cfg.num_workers
        )