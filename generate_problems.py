#%%
import torch
import numpy as np
import pandas as pd
import pickle
from solver import solve_job_shop
from pathlib import Path

#%%
# CONFIG
N = 50  # total instances to create
M = 10  # maximum number of machines for each instance
J = 10  # maximum number of jobs for each instance
T = 10  # maximum number of tasks for each job
D = 20  # maximum duration for each task

min_J = 3 # ensure that we always have more than 2 jobs
min_T = 3 # ensures that we always have more than 2 tasks for each job
min_M = 3 # ensures that we always have more than 2 machines

solutions_dir = Path('solutions')
problems_dir = Path('problems')
solutions_dir.mkdir(exist_ok=True)
problems_dir.mkdir(exist_ok=True)

#%%
problem_tensor_all = []
problem_tuple_all = []
data_view = []
for i in range(N):
    j = np.random.randint(min_J, J)
    t = np.random.randint(min_T, T) 
    m = np.random.randint(min_M, M)
    
    print(f"Generating problem for case {i} has {m} machines and (jobs, tasks): ({j}, {t})")
    machines = torch.randint(low=0, high=m, size=(j,t))
    durations = torch.randint(low=0, high=D, size=(j,t))

    problem_tensor = torch.stack((machines, durations), dim=2)
    problem_tuple = [[(machines[job_idx, task_idx].item(), durations[job_idx, task_idx].item()) for task_idx in range(t)] for job_idx in range(j)]  # required as tuple input for Google OR tools
    
    problem_tensor_all.append(problem_tensor)
    problem_tuple_all.append(problem_tuple)

    job_label = [f'job_{i}' for i in np.arange(j)]
    tasks_label = [f'task_{i}' for i in np.arange(t)]
    df = pd.DataFrame(problem_tuple, index=job_label, columns=tasks_label)

    file_name = f'case_{i}_m{m}_j{j}_t{t}'
    df.to_csv(problems_dir / f"{file_name}.csv")

    solution_output = solve_job_shop(problem_tuple)
    with open(solutions_dir / f"{file_name}.txt", 'w') as file:
        file.write(solution_output)


#%%
problem_tensor_all_name = "tensors_all.pkl"
with open(problem_tensor_all_name, 'wb') as file: # 'wb' for write binary
    pickle.dump(problem_tensor_all, file)

#%%
problem_tuple_all_name = "tuples_all.pkl"
with open(problem_tuple_all_name, 'wb') as file: # 'wb' for write binary
    pickle.dump(problem_tuple_all, file)
# %%
