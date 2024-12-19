import os
import streamlit as st
import requests
import io
from PIL import Image
from typing import Optional
import logging
import mimetypes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GATEWAY_SERVICE = os.getenv("GATEWAY_SERVICE")
if not GATEWAY_SERVICE:
    raise ValueError("GATEWAY_SERVICE environment variable is not set")

def process_prediction(response: requests.Response) -> Optional[str]:
    """Process the prediction response and handle errors."""
    try:
        if response.status_code == 200:
            result = response.json()
            return f"Prediction: {result['prediction']}"
        else:
            logger.error(f"Prediction failed with status code: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error processing prediction: {str(e)}")
        return None

def main():
    st.title("Image Classification Service")
    
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload an image file (JPG, JPEG, or PNG)"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        if st.button('Predict', help="Click to get prediction for the uploaded image"):
            with st.spinner('Processing image...'):
                # Get the file's content type based on its extension
                file_type = mimetypes.guess_type(uploaded_file.name)[0]
                if file_type is None:
                    file_type = 'application/octet-stream'
                
                files = {
                    'file': (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        file_type
                    )
                }
                
                try:
                    response = requests.post(
                        f'{GATEWAY_SERVICE}/predict',
                        files=files,
                        timeout=30
                    )
                    result = process_prediction(response)
                    if result:
                        st.success(result)
                    else:
                        st.error("Error occurred during prediction")
                except requests.exceptions.RequestException as e:
                    logger.error(f"Request failed: {str(e)}")
                    st.error("Failed to connect to the gateway service")

if __name__ == "__main__":
    main()