from ts.torch_handler.base_handler import BaseHandler
import torch
import json
import logging
import numpy as np
from model.data import Data
from model.architecture import DenseNet

logger = logging.getLogger(__name__)

class TorchServeHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.model = DenseNet(in_channels=3, num_classes=10)
        state_dict = torch.load(f"{model_dir}/model.pt", map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.initialized = True
    
    def preprocess(self, data):
        images = []
        for row in data:
            image = torch.tensor(row.get("data")).float()
            images.append(image)
        
        return torch.cat(images, 0).to(self.device)
    
    def inference(self, X):
        with torch.no_grad():
            outputs = self.model(X)
            predictions = torch.argmax(outputs, dim=1)
        return predictions

    def postprocess(self, predictions):
        return [{"prediction": pred.item() for pred in predictions}]

