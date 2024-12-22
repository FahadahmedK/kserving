import os
import json
from typing import List, Dict
from ts.torch_handler.base_handler import BaseHandler
import torch
import logging
from model.architecture import DenseNet

logger = logging.getLogger(__name__)

class TorchServeHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False
        logger.info("Initializing TorchServeHandler")

    def initialize(self, ctx):
        logger.info("Starting model initialization")
        try:
            properties = ctx.system_properties
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            model_dir = properties.get("model_dir")
            manifest = ctx.manifest
            logger.debug(f"Model directory: {model_dir}")

            self.model = DenseNet(
                in_channels=3,
                num_classes=10
            )
            logger.debug("DenseNet model instance created")

            serialized_file = manifest['model']['serializedFile']
            model_pt_path = os.path.join(model_dir, serialized_file)
            logger.debug(f"Loading model from path: {model_pt_path}")
            
            state_dict = torch.load(model_pt_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully on: {self.device}")
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}", exc_info=True)
            raise

    def preprocess(self, data: List) -> torch.Tensor:
        logger.debug(f"Preprocessing batch of size: {len(data)}")
        try:
            images = []
            for i, row in enumerate(data):
                # Log first few items for debugging
                if i < 3:
                    logger.debug(f"Sample {i} shape: {len(row.get('data', []) or row.get('body', []))}")
                
                image = torch.tensor(row.get("data") or row.get("body")).float()
                images.append(image)
            
            batch = torch.cat(images, 0).to(self.device)
            logger.debug(f"Preprocessed batch shape: {batch.shape}")
            return batch
            
        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}", exc_info=True)
            raise
    
    def inference(self, image_batch: torch.Tensor) -> torch.Tensor:
        logger.debug(f"Running inference on batch of shape: {image_batch.shape}")
        try:
            with torch.no_grad():
                outputs = self.model(image_batch)
                predictions = torch.argmax(outputs, dim=1)
                logger.debug(f"Generated predictions shape: {predictions.shape}")
                return predictions
                
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}", exc_info=True)
            raise

    def postprocess(self, predictions: torch.Tensor) -> List[Dict[str, int]]:
        logger.debug(f"Postprocessing predictions of shape: {predictions.shape}")
        try:
            results = [{"prediction": pred.item()} for pred in predictions]
            logger.debug(f"Postprocessed results for first 3 samples: {results[:3]}")
            return results
            
        except Exception as e:
            logger.error(f"Error during postprocessing: {str(e)}", exc_info=True)
            raise

    def handle(self, data, context):
        """
        Custom handle method to ensure model is initialized
        """
        logger.info("Starting request handling")
        try:
            if not self.initialized:
                logger.info("Model not initialized, initializing now")
                self.initialize(context)
                
            if data is None:
                logger.warning("Received null data")
                return None
            
            logger.info(f"Processing batch of size: {len(data)}")
            
            # Preprocess
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if start_time:
                start_time.record()
            
            image_batch = self.preprocess(data)
            
            # Inference
            predictions = self.inference(image_batch)
            
            # Postprocess
            results = self.postprocess(predictions)
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                logger.info(f"Total processing time: {start_time.elapsed_time(end_time):.2f}ms")
            
            logger.info("Request handled successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error handling request: {str(e)}", exc_info=True)
            raise