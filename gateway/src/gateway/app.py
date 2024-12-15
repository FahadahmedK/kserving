import os
from fastapi import FastAPI, HTTPException, UploadFile, File
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import requests
import numpy as np

from model.data import Data

MODEL_SERVER = os.getenv('MODEL_SERVER')

app = FastAPI(title="Gateway to Image Classifier")

async def process_image(image_bytes):

    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    tensor = Data.test_transform(image)
    return tensor.unsqueeze(0).numpy()


@app.post('/predict')
async def predict(file: UploadFile = File()):
    
    image_bytes = file.read()
    processed_image = await process_image(image_bytes)
    headers = {'Content-Type': 'application/json'}
    model_endpoint = f'{MODEL_SERVER}/predictions/densenet'
    response = requests.post(
        url=model_endpoint,
        json={'data': processed_image.tolist()},
        headers=headers
    )

    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
