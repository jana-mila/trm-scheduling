#%%
import torch 
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Optional
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path

#%%
# CONFIG
problems_dir = 'data/problems'
solutions_dir = 'data/solutions'

#%%
class CustomJobShopDataset(Dataset):
    def __init__(self, problems_dir, solutions_dir, transform=None):
        self.problems_dir = Path(problems_dir)
        self.solutions_dir = Path(solutions_dir)
        self.transform = transform

        # ensure both directories contain same number of files
        self.num_problems = sum(1 for f in self.problems_dir.iterdir() if f.is_file())
        self.num_solutions = sum(1 for f in self.solutions_dir.iterdir() if f.is_file())
        assert self.num_problems == self.num_solutions, (
            f"Number of problems {num_problems} does not match \nnumber of solutions {num_solutions}."
        )

        self.problems_files = sorted(self.problems_dir.iterdir())
        self.solutions_files = sorted(self.solutions_dir.iterdir())

    def __len__(self):
        return self.num_problems

    def __getitem__(self, idx):
        problem = torch.load(self.problems_files[idx])
        solution = torch.load(self.solutions_files[idx])
        sample = {'problem': problem, 'solution': solution}
        if self.transform:
            sample = self.transform(sample)
        return sample


def collate_fn(batch):
    problems = [item['problem'].view(-1, 2) for item in batch]
    solutions = [item['solution'].view(-1) for item in batch]

    pad_problems = pad_sequence(problems, batch_first=True)   # [B, max_S, 2]
    pad_solutions = pad_sequence(solutions, batch_first=True) # [B, max_S]
    mask = torch.arange(pad_solutions.size(1))[None, :] < torch.tensor([s.size(0) for s in solutions])[:, None]
    
    return {'problem': pad_problems, 'solution': pad_solutions, 'mask': mask}

#%%
def main():
    #%%
    dataset = CustomJobShopDataset(problems_dir, solutions_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    for batch in dataloader:
        print(batch['problem'].shape, batch['solution'].shape)
        break


#%%


#%%

#%%
if __name__ == '__main__':
    main()