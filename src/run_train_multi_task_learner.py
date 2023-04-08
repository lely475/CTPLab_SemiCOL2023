import argparse
import shutil

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from dataloader.multi_task_dataset import MultiTaskDataModule
from multi_task_learning.custom_callbacks import GalleryLogger, MetricsLogger
from multi_task_learning.multi_task_learner import MultiTaskLearner
from utils.config_templates.train import TrainConfig
from utils.utils import create_output_folder, set_random_seeds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multi task learner")
    parser.add_argument(
        "--o", help="Defines the output directory.", type=str, required=True
    )

    parser.add_argument(
        "--arm",
        help="Specifies, whether training is for Arm 1 or Arm 2. Can have value 1 or 2 respectively",
        type=int,
        required=True,
    )

    args = parser.parse_args()
    config_file = "configs/run_train_multi_task_learner.yaml"
    if args.arm != 1 and args.arm != 2:
        raise ValueError(
            f"Argument --arm can only have values 1 and 2, for Arm 1 and Arm 2 configuration, not {args.arm}."
        )

    with open(config_file, "r") as yaml_file:
        config = TrainConfig.from_dict(yaml.safe_load(yaml_file))

    config.data_augmentation.img_stat_ref_images = f"references_arm{args.arm}"

    # Create output folder, copy config file
    config.paths.output_path = args.o
    config.paths.output_path = create_output_folder(config)
    shutil.copy(config_file, config.paths.output_path)

    # Init seeds, cuda, dataloader and model
    set_random_seeds(config.experiment.seed)
    torch.cuda.empty_cache()
    dm = MultiTaskDataModule(config)
    model = MultiTaskLearner(config)

    # Init trainer and train
    trainer = Trainer(
        accelerator="gpu",
        max_epochs=config.experiment.epochs,
        check_val_every_n_epoch=1,
        multiple_trainloader_mode="min_size",
        logger=TensorBoardLogger(
            save_dir=f"{config.paths.output_path}/tb_log", name="", version=""
        ),
        callbacks=[MetricsLogger(), GalleryLogger()],
        limit_val_batches=0 if config.experiment.full_set else None,
        num_sanity_val_steps=0 if config.experiment.full_set else 2,
    )
    trainer.fit(model, dm)
