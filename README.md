# trm-scheduling

*By: Jana Osea*

*Updated as of November 21, 2025*


**Goal:** Solve Job Shop scheduling with the objective to minimize makespan (total length of time from start of earliest job and end of last job) while adhering to precedence and disjunctive constraints using supervised learning technique particularly the [Tiny Recursive Model](https://arxiv.org/abs/2510.04871) proposed by Alexia.

## What's in this directory

```
├── generate_data.py
├── train.py
├── configs/                        # Needs improvement to manage versions
├── data/                           # Contains v1 and v2 based on number of inputs
├── src/                            # Contains logic for PyTorch lightning
│   ├── datamodule.py   
│   ├── model.py
│   ├── TinyRecursiveModels/        # Modifications of TRM code
│   └── utils.py
├── scripts/                        # Contains script to submit job training
```

## Generate Data

Use `generate_data.py` to generate the problems and the solutions specifying number of problem instances, number of machines, jobs, and tasks for each instance and setting the maximum duration for each task. The data will be stored in `data` directory. No configs files exist, just have to modify in the script. 


## Train model

There are multiple config files to choose. To send a training job on the cluster use the following code where you can choose which config you want to train.

```
sbatch scripts/submit_training.sh --config-name default
```

If you are on an interactive node and want to train, use the following

```
python train.py --config-name default
```

Note: In order to change the version number, you need to edit the `train.py` hydra. Since the addition of new versions was recent, I haven't gotten that change to centralize the configs better.

## Inference model

Since the refactor, the old checkpoints are not compatible with the configs right now. But the idea is

```
with initialize(version_base = "1.3", config_path = "configs"):
    cfg = compose(config_name="default.yaml")
datamodule = JobShopDataModule(config = cfg, batch_size = cfg.batch_size, max_seq_len = cfg.model.trm_model.seq_len)
datamodule.setup()
val_dl = datamodule.val_dataloader()

checkpoint_path = "some checkpoint path"
model = TinyRecursiveModelJobShop.load_from_checkpoint(checkpoint_path)
model.eval()
model.freeze()

with torch.no_grad():
    carry = model.model.initial_carry(batch)
    carry = carry.to(model.device)
    for step in range(cfg.model.trm_model.halt_max_steps):
        carry, output = model(carry = carry, batch = batch)
        y_hat = output['logits']    # Predicted values
        y_true = batch['labels']    # True values
        mask = batch['mask']
```
