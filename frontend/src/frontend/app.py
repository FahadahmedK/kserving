import os
import streamlit as st
import requests
import io
from PIL import Image


GATEWAY_SERVICE = os.getenv("GATEWAY_SERVICE")

def main():
    st.title("Image Classification Service")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Predict'):
            # Send image to gateway service
            files = {'file': uploaded_file.getvalue()}
            response = requests.post(f'{GATEWAY_SERVICE}/predict', files=files)
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"Prediction: {result['prediction']}")
            else:
                st.error("Error occurred during prediction")

if __name__ == "__main__":
    main()