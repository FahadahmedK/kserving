# Gateway Service (app.py)
import os
import io
import requests
import numpy as np
from typing import Dict, Any
import logging
import traceback
from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.responses import JSONResponse
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_SERVICE = os.getenv('MODEL_SERVICE')
if not MODEL_SERVICE:
    raise ValueError("MODEL_SERVICE environment variable is not set")

app = FastAPI(
    title="Gateway to Image Classifier",
    description="API gateway for image classification service",
    version="1.0.0"
)

def resize_with_aspect_ratio(image: np.ndarray, target_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    if height < width:
        new_height = target_size
        new_width = int((width / height) * target_size)
    else:
        new_width = target_size
        new_height = int((height/width) * target_size)

    img = Image.fromarray(image)
    img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)
    return np.array(img)

def center_crop(image: np.ndarray, crop_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    start_y = (height - crop_size) // 2
    start_x = (width - crop_size) //2
    return image[start_y:start_y + crop_size, start_x: start_x+ crop_size]

async def process_image(image_bytes: bytes) -> np.ndarray:
    """Process image bytes into normalized tensor."""
    
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = np.array(image)

        # Resize
        image = resize_with_aspect_ratio(image, 256)
        
        # Center crop
        image = center_crop(image, 224)

        # normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # normalize dimensionwise 
        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)

        return image

    except Exception as e:
        logger.exception("Image processing error") 
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format or corrupted file"
        ) from e

@app.post('/predict', response_model=Dict[str, Any])
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """
    Predict image classification using the ML model service.
    
    Args:
        file: Uploaded image file
    
    Returns:
        JSON response with prediction results
    """
    print(file)
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file uploaded"
        )

    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )
    
    try:
        image_bytes = await file.read()
        processed_image = await process_image(image_bytes)
        
        headers = {'Content-Type': 'application/json'}
        model_endpoint = f'{MODEL_SERVICE}/predictions/densenet'
        
        response = requests.post(
            url=model_endpoint,
            json={'data': processed_image.tolist()},
            headers=headers,
            timeout=30  # Add timeout
        )
        
        if response.status_code != 200:
            logger.error(f"Model service error: {response.status_code}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Error from model service"
            )
            
        return JSONResponse(content=response.json())
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request to model service failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service unavailable"
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}