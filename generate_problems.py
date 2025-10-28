#%%
import torch
import numpy as np
import pandas as pd
import pickle
from solver import solve_job_shop

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

#%%
data_tensors = []
data_tuples = []
data_view = []
for i in range(N):
    j = np.random.randint(min_J, J)
    t = np.random.randint(min_T, T) 
    m = np.random.randint(min_M, M)
    
    print(f"Case {i} has {m} machines and (jobs, tasks): ({j}, {t})")
    machines = torch.randint(low=0, high=m, size=(j,t))
    durations = torch.randint(low=0, high=D, size=(j,t))

    record_tensor = torch.stack((machines, durations), dim=2)
    record_list_of_tuples = [[(machines[job_idx, task_idx].item(), durations[job_idx, task_idx].item()) for task_idx in range(t)] for job_idx in range(j)]
    
    data_tensors.append(record_tensor)
    data_tuples.append(record_list_of_tuples)

    job_label = [f'job_{i}' for i in np.arange(j)]
    tasks_label = [f'task_{i}' for i in np.arange(t)]
    df = pd.DataFrame(record_list_of_tuples, index=job_label, columns=tasks_label)

    file_name = f'case_{i}_m{m}_j{j}_t{t}'
    df.to_csv(f"problems/{file_name}.csv")

    solution_output = solve_job_shop(record_list_of_tuples)
    with open(f"solutions/{file_name}.txt", 'w') as file:
        file.write(solution_output)


#%%
data_tensors_name = "tensors_all.pkl"
with open(data_tensors_name, 'wb') as file: # 'wb' for write binary
    pickle.dump(data_tensors, file)

#%%
data_tuples_name = "tuples_all.pkl"
with open(data_tuples_name, 'wb') as file: # 'wb' for write binary
    pickle.dump(data_tuples, file)
# %%
