# %%
import torch
import numpy as np
import pandas as pd
import multiprocessing
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
import sys

from src.utils import solve_job_shop


# =============================
# CONFIGURATION
# =============================

@dataclass
class Config:
    VERSION: int = 2                # Choices of only 1 or 2, difference is in problem output. See generate_problem for more details
    PURPOSE: str = "train"          # Choices of 'train', 'val', 'test'
    N: int = 500                    # Number of job shop problems to generate
    TIMEOUT_SECONDS: int = 60       # Seconds for solver to find solution before moving on to the next

    # Dimensional bounds
    M: int = 20                     # Maximum number of machines
    J: int = 20                     # Maximum number of jobs per problems
    T: int = 20                     # Maximum number of taks per job
    D: int = 10                     # Maximum duration time units

    MIN_J: int = 3                  # Minimums
    MIN_T: int = 3
    MIN_M: int = 3
    MIN_D: int = 1

cfg = Config()


# =============================
# DIRECTORY SETUP
# =============================

def setup_directories(cfg: Config) -> tuple[Path, Path, Path]:
    subset = f"v{cfg.VERSION}/{cfg.PURPOSE}/{cfg.N}"
    base = Path.cwd() / "data" / subset

    for sub in ["solutions", "problems", "cases"]:
        (base / sub).mkdir(parents=True, exist_ok=True)

    print(f"Saving data to: {base}")
    return base / "solutions", base / "problems", base / "cases"


# =============================
# SOLVER WRAPPER
# =============================

def solve_job_shop_process(problem_input, queue):
    """Runs solve_job_shop in a separate process and returns results through queue."""
    try:
        solution_txt, solution = solve_job_shop(problem_input)
        queue.put(("success", solution_txt, solution.tolist()))
    except Exception as e:
        queue.put(("error", str(e), None))
        sys.exit(1)


# =============================
# PROBLEM GENERATION
# =============================

def generate_problem(cfg: Config):
    j = np.random.randint(cfg.MIN_J, cfg.J + 1)
    t = np.random.randint(cfg.MIN_T, cfg.T + 1)
    m = np.random.randint(cfg.MIN_M, cfg.M + 1)

    machines = torch.randint(0, m, (j, t))
    durations = torch.randint(cfg.MIN_D, cfg.D + 1, (j, t))

    if cfg.VERSION == 1:
        # Version 1: problem tensor is [machines, durations]
        problem = torch.stack((machines, durations), dim=2)
    elif cfg.VERSION == 2:
        # Version 2: problem tensor is [job_index, task_index, machines, durations]
        js = torch.arange(j).view(-1, 1).repeat(1, t)
        ts = torch.arange(t).repeat(j, 1)
        problem = torch.stack((js, ts, machines, durations), dim=2)

    # Convert into solver-friendly format: list of jobs
    problem_input = [
        [(machines[job, task].item(), durations[job, task].item()) for task in range(t)]
        for job in range(j)
    ]
    return problem, problem_input, (j, t, m)


# =============================
# SAVE ARTIFACTS
# =============================

def save_results(i, problem, solution, dims, solution_txt, dirs):
    solutions_dir, problems_dir, cases_dir = dirs
    j, t, m = dims

    file = f"case_{i}_m{m}_j{j}_t{t}"

    torch.save(problem, problems_dir / f"{file}.pt")
    torch.save(solution, solutions_dir / f"{file}.pt")

    df = pd.DataFrame(
        [[*pair] for job in problem.tolist() for pair in job],  # flatten format
    )

    case_text = f"\nProblem (Version {cfg.VERSION})\n{df}\n\n{solution_txt}"

    with open(cases_dir / f"{file}.txt", "w") as f:
        f.write(case_text)

# =============================
# MAIN LOOP
# =============================

if __name__ == '__main__':
    solutions_dir, problems_dir, cases_dir = setup_directories(cfg)
    print(f"Generating dataset N={cfg.N}")

    for i in tqdm(range(cfg.N)):
        problem, problem_input, dims = generate_problem(cfg)

        # run solver in separate process
        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=solve_job_shop_process, args=(problem_input, q))
        p.start()
        p.join(timeout=cfg.TIMEOUT_SECONDS)

        if p.is_alive():
            p.terminate()
            p.join()
            continue  # timeout occurred

        if q.empty():
            continue

        status, solution_txt, solution = q.get()
        if status == "error":
            continue

        save_results(i, problem, torch.tensor(solution), dims, solution_txt, (solutions_dir, problems_dir, cases_dir))