import torch
import torch.nn as nn
from ray.train.torch import (
    TorchTrainer,
    TorchConfig
)
from ray.data import Dataset
from ray.train import (
    ScalingConfig, 
    RunConfig,
    DataConfig,
    Checkpoint,
    CheckpointConfig
)

from model.data import Dataset
from model.architecture import DenseNet

def train_step(
    ds: Dataset,

)

def train_func(config: dict):




trainer =