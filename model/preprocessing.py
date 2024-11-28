import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os

from base import PipelineStep

class Preprocessing:
    
    def __init__(self, input_path, output_path):

        self.input_path = input_path
        self.output_path = output_path