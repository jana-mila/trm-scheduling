#%%
import torch
import numpy as np
import pandas as pd
import pickle
import multiprocessing
from src.utils import solve_job_shop
from pathlib import Path
from tqdm import tqdm
import sys

#%%
# CONFIG
subset = "v2/train/1000"
N = 1000
M = 20
J = 20
T = 20
D = 10
TIMEOUT_SECONDS = 60

min_J = 3
min_T = 3
min_M = 3
min_D = 1

# Determine base directory
if '__file__' in locals():
    data_dir = Path(__file__).resolve().parent / "data" / f'{subset}'
else:
    data_dir = Path.cwd() / "data" / f'{subset}'

data_dir.mkdir(parents=True, exist_ok=True)

solutions_dir = Path(data_dir / 'solutions')
problems_dir = Path(data_dir / 'problems')
cases_dir = Path(data_dir / 'cases')

solutions_dir.mkdir(exist_ok=True, parents=True)
problems_dir.mkdir(exist_ok=True, parents=True)
cases_dir.mkdir(exist_ok=True, parents=True)

# --- Multiprocessing Wrapper ---
def solve_job_shop_with_queue(problem_input, queue):
    """Run solve_job_shop in its own process and return via Queue."""
    try:
        solution_txt, solution = solve_job_shop(problem_input)

        # Convert tensor to Python-native list so queue does not break
        if isinstance(solution, torch.Tensor):
            solution = solution.tolist()

        queue.put(("success", solution_txt, solution))
    except Exception as e:
        queue.put(("error", str(e), None))
        sys.exit(1)

#%%
# MAIN LOOP
for i in tqdm(range(N)):

    j = np.random.randint(min_J, J + 1)
    t = np.random.randint(min_T, T + 1)
    m = np.random.randint(min_M, M + 1)

    print(f"Case {i}: {m} machines, {j} jobs, {t} tasks/job")

    # Generate assigned machines for each job task
    machines = torch.randint(low=0, high=m, size=(j, t))
    
    # Generate durations for that corresponding job taks
    durations = torch.randint(low=min_D, high=D + 1, size=(j, t))
    
    # Generate the job and task assignment label
    ts = torch.empty((j,t))
    js = torch.empty((j,t))
    for x in range(j):
        for y in range(t):
            ts[x,y] = y
            js[x,y] = x
    
    problem = torch.stack((js, ts, machines, durations), dim=2)
    problem_input = [
        [
            (machines[job_idx, task_idx].item(), durations[job_idx, task_idx].item())
            for task_idx in range(t)
        ]
        for job_idx in range(j)
    ]

    # --- TIMEOUT LOGIC START ---
    result_queue = multiprocessing.Queue()
    solver_process = multiprocessing.Process(
        target=solve_job_shop_with_queue,
        args=(problem_input, result_queue)
    )

    solver_process.start()
    solver_process.join(timeout=TIMEOUT_SECONDS)

    # If still alive, kill and skip
    if solver_process.is_alive():
        print(f"🛑 TIMEOUT on case {i}: exceeded {TIMEOUT_SECONDS} sec. Skipping.")
        solver_process.terminate()
        solver_process.join()
        continue

    # Retrieve from queue if available
    if not result_queue.empty():
        status, solution_txt, solution = result_queue.get()

        if status == "error":
            print(f"⚠️ Error solving case {i}: {solution_txt}. Skipping.")
            continue

        # Convert list back to tensor
        solution = torch.tensor(solution)

    else:
        print(f"⚠️ No result returned for case {i}. Skipping.")
        continue

    # --- Save solution ---
    file_name = f"case_{i}_m{m}_j{j}_t{t}"

    torch.save(problem, problems_dir / f"{file_name}.pt")
    torch.save(solution, solutions_dir / f"{file_name}.pt")

    problem_df = pd.DataFrame(
        problem_input,
        index=[f"job_{i}" for i in range(j)],
        columns=[f"task_{i}" for i in range(t)]
    )

    case_txt = f"\nProblem\n{problem_df.to_string()}\n\n{solution_txt}"

    with open(cases_dir / f"{file_name}.txt", "w") as f:
        f.write(case_txt)

