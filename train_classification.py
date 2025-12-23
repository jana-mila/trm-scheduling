import hydra
import os
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.datamodule_classification import JobShopDataModule
from src.model_classification import TinyRecursiveModelJobShop

@hydra.main(version_base=None, config_path="configs/classification", config_name="default")
def train(cfg: DictConfig):

    print(f"Configuration:/n{OmegaConf.to_yaml(cfg)}")
    
    print("Initializing DataModule...")
    datamodule = JobShopDataModule(config = cfg, batch_size = cfg.batch_size, max_seq_len = cfg.model.trm_model.seq_len)
    datamodule.setup()
    batch = next(iter(datamodule.val_dataloader()))

    print("Initializing LightningModule")
    model = TinyRecursiveModelJobShop(config = cfg.model.trm_model, lr = cfg.lr, act_loss_weight = cfg.act_loss_weight)
    print(model)
    
    wandb_logger = WandbLogger(
        project = cfg.project_name,
        name = cfg.run_name, 
        config = OmegaConf.to_container(cfg, resolve = True)
    )

    project_root = hydra.utils.get_original_cwd()
    run_checkpoint_dir = os.path.join(project_root, cfg.checkpoint_dir, cfg.run_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_checkpoint_dir,
        monitor="val_loss",
        mode="min",
        filename=f"{cfg.run_name}-{{epoch:02d}}-{{val_loss:.4f}}",
        save_top_k=2
    )

    print("Initializing Trainer...")
    trainer = pl.Trainer(
        **cfg.trainer, # Loads max_epochs, accelerator, etc.
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )

    # 5. Start Training
    print("Starting training...")
    trainer.fit(
        model=model, 
        datamodule=datamodule, 
        ckpt_path=cfg.resume_from # PL handles resume logic
    )


if __name__ == "__main__":
    train()