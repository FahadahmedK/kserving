import os
from pathlib import Path
import tempfile
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel.distributed import DistributedDataParallel
import mlflow
from ray.train.torch import (
    TorchTrainer,
    TorchConfig
)
import ray
from ray.data import Dataset
from ray.train import (
    ScalingConfig, 
    RunConfig,
    DataConfig,
    Checkpoint,
    CheckpointConfig
)
from ray.air.integrations.mlflow import MLflowLoggerCallback

from model.data import Data
from model.architecture import DenseNet


# Config MLflow
EFS_DIR = Path(f"{Path(__file__).parent.parent.absolute()}/rayruns")

Path(EFS_DIR).mkdir(parents=True, exist_ok=True)

MODEL_REGISTRY = Path(f"{EFS_DIR}/mlflow")
Path(MODEL_REGISTRY).mkdir(parents=True, exist_ok=True)
MLFLOW_TRACKING_URI = "file://" + str(MODEL_REGISTRY.absolute())
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def one_train_step(
    ds: Dataset,
    batch_size :int,
    model: nn.Module,
    loss_fn: torch.nn.modules.loss._WeightedLoss,
    optimizer: torch.optim.Optimizer
) -> float:
    
    model.train()
    loss = 0.0
    ds_generator = ds.iter_torch_batches(batch_size=batch_size, local_shuffle_buffer_size=1000)
    for i, batch in enumerate(ds_generator):
        optimizer.zero_grad()
        imgs, labels = batch["images"], batch["labels"]
        if i == 0:
            print(labels)
        logits = model(imgs) # (b, 10)
        batch_loss = loss_fn(logits, labels)
        batch_loss.backward()
        optimizer.step()
        loss += (batch_loss.detach().item() - loss) / (i+1)

    return loss

def one_eval_step(
    ds: Dataset,
    batch_size :int,
    model: nn.Module,
    loss_fn: torch.nn.modules.loss._WeightedLoss
):
    model.eval()
    loss = 0.0
    y_trues, y_preds = [], []
    ds_generator = ds.iter_torch_batches(batch_size=batch_size, local_shuffle_buffer_size=500)
    with torch.inference_mode():
        for i, batch in enumerate(ds_generator):
            imgs, labels = batch["images"], batch["labels"]
            logits = model(imgs)
            batch_loss = loss_fn(logits, labels)
            loss += (batch_loss.detach().item() - loss) / (i+1)
            y_trues.extend(labels.cpu().numpy())
            y_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
    return loss, np.vstack(y_trues), np.vstack(y_preds)



def train_func_per_worker(config: dict):

    num_epochs = config["num_epochs"]
    lr = config["lr"]
    lr_factor = config["lr_factor"]
    lr_patience = config["lr_patience"]
    batch_size = config["batch_size"]
    num_classes = config["num_classes"]

    train_ds = ray.train.get_dataset_shard("train")
    val_ds = ray.train.get_dataset_shard("val")

    model = DenseNet(in_channels=3, num_classes=num_classes)
    model = ray.train.torch.prepare_model(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=lr_factor, patience=lr_patience)
    num_workers = ray.train.get_context().get_world_size()
    batch_size_per_worker = batch_size # // num_workers

    for epoch in range(num_epochs):

        train_loss = one_train_step(
            ds=train_ds,
            batch_size=batch_size_per_worker,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer
        )

        val_loss, y_trues, y_preds = one_eval_step(
            ds=val_ds,
            batch_size=batch_size_per_worker,
            model=model,
            loss_fn=loss_fn
        )

        accuracy = (y_trues == y_preds).sum() / len(y_preds)

        weight = 0.5
        weighted_checkpoint_metric = - weight * 2 * accuracy + (1-weight) * val_loss 

        # scheduler.step(val_loss)

        with tempfile.TemporaryDirectory() as dp:
            model.save(dp=dp)

            metrics = dict(epoch=epoch, accuracy=accuracy, train_loss=train_loss, val_loss=val_loss, weighted_checkpoint_metric=weighted_checkpoint_metric)
            checkpoint = Checkpoint.from_directory(dp)
            ray.train.report(metrics, checkpoint=checkpoint)

        

def main():

    scaling_config = ScalingConfig(
        num_workers=1,
        use_gpu=True,
        # resources_per_worker={
        #     "CPU": 4,           
        #     "GPU": 1.0           
        # },
        # placement_strategy="PACK",
    )

    checkpoint_config = CheckpointConfig(
        num_to_keep=5,
        checkpoint_score_attribute="val_loss",
        checkpoint_score_order="min"
    )

    mlflow_callback = MLflowLoggerCallback(
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name="clothing",
        save_artifact=True,
    )

    run_config = RunConfig(
        callbacks=[mlflow_callback], 
        checkpoint_config=checkpoint_config, 
        storage_path=EFS_DIR
        )
    

    train_data = Data("/home/fahad/study/kserving/data/train")
    train_ds, class_to_idx = train_data.create_dataset()
    val_data = Data("/home/fahad/study/kserving/data/validation", train=False)
    val_ds, _ = val_data.create_dataset()    

    train_loop_config = {
        "num_epochs": 200,
        "batch_size": 32,
        "lr": 1.0e-4,
        "lr_factor": 0.1,
        "lr_patience": 0.3,
        "num_classes": len(class_to_idx)
    }

    train_ds = train_ds.materialize()
    val_ds = val_ds.materialize()

    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        run_config=run_config,
        datasets={"train": train_ds, "val": val_ds},
        torch_config=TorchConfig(
            backend="nccl"
        )
    )
    results = trainer.fit()

    best_checkpoint = results.checkpoint

if __name__ == "__main__":

    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    main()


