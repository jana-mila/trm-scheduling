#%%
import torch
import numpy as np
import pandas as pd
import pickle
from solver import solve_job_shop
from pathlib import Path
from tqdm import tqdm

#%%
# CONFIG
N = 1000  # total instances to create
M = 20  # maximum number of machines for each instance
J = 20 # maximum number of jobs for each instance
T = 20  # maximum number of tasks for each job
D = 10  # maximum duration for each task

min_J = 3 # ensure that we always have more than 2 jobs
min_T = 3 # ensures that we always have more than 2 tasks for each job
min_M = 3 # ensures that we always have more than 2 machines
min_D = 1 # ensure that we always nonzero task duration

solutions_dir = Path('data/solutions')
problems_dir = Path('data/problems')
cases_dir = Path('data/cases')

solutions_dir.mkdir(exist_ok=True, parents=True)
problems_dir.mkdir(exist_ok=True, parents=True)
cases_dir.mkdir(exist_ok=True, parents=True)

#%%
for i in tqdm(range(N)):

    j = np.random.randint(min_J, J)
    t = np.random.randint(min_T, T) 
    m = np.random.randint(min_M, M)

    

    print(f'case {i} with {m} machines, {j} jobs, {t} tasks each job')    
    machines = torch.randint(low=0, high=m, size=(j,t))
    durations = torch.randint(low=0, high=D, size=(j,t))

    problem = torch.stack((machines, durations), dim=2)
    problem_input = [[(machines[job_idx, task_idx].item(), durations[job_idx, task_idx].item()) for task_idx in range(t)] for job_idx in range(j)]  # required as tuple input for Google OR tools
    solution_txt, solution = solve_job_shop(problem_input)

    file_name = f'case_{i}_m{m}_j{j}_t{t}'
    
    torch.save(problem, problems_dir / f'{file_name}.pt')
    torch.save(solution, solutions_dir / f'{file_name}.pt')

    # save each case in more readable format
    problem_df = pd.DataFrame(problem_input, index=[f'job_{i}' for i in np.arange(j)], columns=[f'task_{i}' for i in np.arange(t)])
    case_txt = f'\n Problem\n {problem_df.to_string()}\n' + solution_txt
    
    with open(cases_dir / f"{file_name}.txt", 'w') as file:
        file.write(case_txt)

# %%
