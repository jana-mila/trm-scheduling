# trm-scheduling

*By: Jana Osea*

*Updated as of January 16, 2026*


**Goal:** Solve Job Shop scheduling with the objective to minimize makespan (total length of time from start of earliest job and end of last job) while adhering to precedence and disjunctive constraints using supervised learning technique particularly the [Tiny Recursive Model](https://arxiv.org/abs/2510.04871) proposed by Alexia. The most recent report with the details about data generation, model adaptation, experiment setup, and results can be found [here](https://drive.google.com/file/d/16dmmM8KtYAyE3qi5UT0QKrFB5-7cBxbf/view?usp=sharing)

## What's in this directory

```
├── configs/                # Training configs
├── data/                   # Data used for training and validation
├── src/                    # Contains logic for PyTorch lightning
│   ├── regression/         # Train, Datamodule, modification of TRM code (for regression)
│   ├── classification/     # Train, Datamodule, modification of TRM code (for classification)
│   └── utils.py            
└── scripts/                # Contains script to submit job training
```

## Generate Data

Use `data/generate_data.py` to generate the problems and the solutions specifying number of problem instances, number of machines, jobs, and tasks for each instance and setting the maximum duration for each task. The data will be stored in `data` directory. No configs files exist, just have to modify in the script. 


## Train model

There are multiple config files to choose from, see the config folder to familiarize. To send a training job on the cluster use the following code where you can choose which config you want to train.

```
sbatch scripts/submit_training.sh -m src.classification.train --config-name <INSERT-NAME>
```

If you are on an interactive node and want to train, use the following

```
python -m src.classification.train --config-name <INSERT-NAME>
```

## Inference model

To obtain the checkpoints, copy the models from the project directory into your current directory using the following terminal command:

```
bash cp -r /home/mila/o/oseaj/projects/trm-scheduling/checkpoints/ .
```

Then substitute the checkpoint path in the Python notebook file `notebooks/inference.ipynb`