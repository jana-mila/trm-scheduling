# trm-scheduling

*By: Jana Osea*

*Updated as of December 23, 2025*


**Goal:** Solve Job Shop scheduling with the objective to minimize makespan (total length of time from start of earliest job and end of last job) while adhering to precedence and disjunctive constraints using supervised learning technique particularly the [Tiny Recursive Model](https://arxiv.org/abs/2510.04871) proposed by Alexia. The most recent report with the details about data generation, model adaptation, experiment setup, and results can be found [here](https://drive.google.com/file/d/16dmmM8KtYAyE3qi5UT0QKrFB5-7cBxbf/view?usp=sharing)

## What's in this directory

```
├── generate_data.py
├── train_classification.py
├── train_regression.py
├── configs/                                   # Needs improvement to manage versions
├── data/                                      # Contains v1 and v2 based on number of inputs
├── src/                                       # Contains logic for PyTorch lightning
│   ├── datamodule.py   
│   ├── model.py
│   ├── TinyRecursiveModels_regression/        # Modifications of TRM code (for regression)
│   ├── TinyRecursiveModels_classification/    # Modifications of TRM code (classification)
│   └── utils.py
├── scripts/                                   # Contains script to submit job training
```

## Generate Data

Use `generate_data.py` to generate the problems and the solutions specifying number of problem instances, number of machines, jobs, and tasks for each instance and setting the maximum duration for each task. The data will be stored in `data` directory. No configs files exist, just have to modify in the script. 


## Train model

There are multiple config files to choose from, see the config folder to familiarize. To send a training job on the cluster use the following code where you can choose which config you want to train.

```
sbatch scripts/submit_training.sh train_classification.py --config-name default
```

If you are on an interactive node and want to train, use the following

```
python train_classification.py --config-name default
```

## Inference model

To obtain the checkpoints, copy the models from the project directory into your current directory using the following terminal command:

```
bash cp -r /home/mila/o/oseaj/projects/trm-scheduling/checkpoints/ .
```

Then substitute the checkpoint path in the Python notebook file `notebooks/inference.ipynb`