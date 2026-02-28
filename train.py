import os
import torch
import random
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
import lightning as L
import wandb

from pathlib import Path
from lightning.pytorch.loggers import WandbLogger
from models.modelzoo import lt_segformerb0, lt_mobilevit_dlv3, lt_lraspp_mv3
from models.MeTU import lt_MeTU
from torch.utils.data import DataLoader

MODEL_NAME = "[MeTU-xxs]-v3"
DATE = "0301"
DATASET_NAME = "VOC2012"
BEST_MODEL_PATH = f"./logs/{DATASET_NAME}/{DATE}/{MODEL_NAME}/best"


if DATASET_NAME == "Cityscapes":
    from datasets.CityScapes import (
        CityScapes,
        TrainTransforms,
        ValTransforms,
        load_weight_sampler,
    )

    train_tf = TrainTransforms()
    val_tf = ValTransforms()

    ds_train = CityScapes(mode="train", transforms=train_tf)
    ds_valid = CityScapes(mode="val", transforms=val_tf)

    CLASSES = 19
    IGNORE_IDX = 255
    PROJECT_NAME = "CityScapes - Semantic Segmentation"

elif DATASET_NAME == "VOC2012":
    from datasets.VOC2012 import (
        VOC2012,
        TrainTransforms,
        ValTransforms,
        load_weight_sampler,
    )

    train_tf = TrainTransforms()
    val_tf = ValTransforms()

    ds_train = VOC2012(mode="train", transforms=train_tf)
    ds_valid = VOC2012(mode="val", transforms=val_tf)

    CLASSES = 21
    IGNORE_IDX = 255
    PROJECT_NAME = "PASCAL VOC 2012 - Semantic segmentation"


def set_seed(seed: int = 333):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    L.seed_everything(seed, workers=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class FinalBestModelSaver(L.pytorch.callbacks.Callback):
    def __init__(self, final_save_dir: str):
        super().__init__()
        self.final_save_dir = Path(final_save_dir)
        self.final_save_dir.mkdir(parents=True, exist_ok=True)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        checkpoint_callback = trainer.checkpoint_callback
        if not checkpoint_callback:
            print("ModelCheckpoint callback not found. Cannot save final best model.")
            return

        best_model_path = checkpoint_callback.best_model_path

        if not best_model_path:
            print("No best model path found (Checkpoints might not have been saved).")
            return

        source_path = Path(best_model_path)

        final_file_name = f"{source_path.name}"
        destination_path = self.final_save_dir / final_file_name

        import shutil

        try:
            shutil.copyfile(source_path, destination_path)
            print(f"\n Final Best Model saved to: {destination_path}")
        except Exception as e:
            print(f"\n Error saving final best model: {e}")


if __name__ == "__main__":
    set_seed(333)
    wandb_logger = WandbLogger(
        project=PROJECT_NAME,
        name=MODEL_NAME,
        log_model=True,
        save_dir=f"./wandb_logs/{DATASET_NAME}/{DATE}/{MODEL_NAME}/",
    )

    if "MeTU" in MODEL_NAME:
        if "-xxs" in MODEL_NAME:
            model = lt_MeTU(
                learning_rate=5e-4,
                model_size="xxs",
                encoder_pretrained=True,
                classes=CLASSES,
                ingnore_index=IGNORE_IDX,
            )
        elif "-xs" in MODEL_NAME:
            model = lt_MeTU(
                learning_rate=5e-4,
                model_size="xs",
                encoder_pretrained=True,
                classes=CLASSES,
                ingnore_index=IGNORE_IDX,
            )
    elif "Segformer" in MODEL_NAME:
        model = lt_segformerb0(
            learning_rate=5e-4, classes=CLASSES, ignore_index=IGNORE_IDX
        )
    elif "MobileViT" in MODEL_NAME:
        if "-xxs" in MODEL_NAME:
            model = lt_mobilevit_dlv3(
                model_size="xxs",
                encoder_pretrained=True,
                learning_rate=5e-4,
                classes=CLASSES,
                ignore_index=IGNORE_IDX,
            )
        elif "-xs" in MODEL_NAME:
            model = lt_mobilevit_dlv3(
                model_size="xs",
                encoder_pretrained=True,
                learning_rate=5e-4,
                classes=CLASSES,
                ignore_index=IGNORE_IDX,
            )
    elif "LRASPP" in MODEL_NAME:
        model = lt_lraspp_mv3(
            learning_rate=5e-4,
            classes=CLASSES,
            encoder_pretrained=True,
            ignore_index=IGNORE_IDX,
        )

    weight_sampler = load_weight_sampler(num_samples=len(ds_train))

    train_loader = DataLoader(
        ds_train,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        worker_init_fn=seed_worker,
        sampler=weight_sampler,
    )

    val_loader = DataLoader(ds_valid, batch_size=16, shuffle=False, num_workers=4)

    trainer = L.Trainer(
        max_epochs=200,
        accelerator="auto",
        devices=1,
        precision=16,
        logger=wandb_logger,
        log_every_n_steps=50,
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                dirpath=f"./logs/{DATASET_NAME}/{DATE}/{MODEL_NAME}",
                filename="{epoch:02d}-{val/mIoU:.3f}",
                monitor="val/mIoU",
                mode="max",
                save_top_k=3,
            ),
            L.pytorch.callbacks.EarlyStopping(
                monitor="val/mIoU", patience=30, mode="max"
            ),
            FinalBestModelSaver(final_save_dir=BEST_MODEL_PATH),
        ],
    )
    trainer.fit(model, train_loader, val_loader)

    trainer.test(model, val_loader)

    wandb.finish()
