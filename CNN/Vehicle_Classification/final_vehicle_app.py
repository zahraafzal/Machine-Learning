import streamlit as st
import numpy as np
from PIL import Image

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
import keras

# Page config
st.set_page_config(page_title="Vehicle Classification", layout="centered")

st.title("Vehicle Classification using CNN")
st.markdown("Upload a vehicle image to classify it")

# Load model
@st.cache_resource
def load_model():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, "cars_cnn_compatible.keras")

        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None

        model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
        return model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Class names
CLASS_NAMES = [
    'airplane', 'ambulance', 'bicycle', 'boat', 'bus',
    'car', 'fire_truck', 'helicopter', 'hovercraft', 'jet_ski',
    'kayak', 'motorcycle', 'rickshaw', 'scooter', 'segway',
    'skateboard', 'tractor', 'truck', 'unicycle', 'van'
]

def preprocess_image(img):
    img = img.resize((128, 128))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Main app
if model is not None:
    
    
    uploaded_file = st.file_uploader("Choose a vehicle image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            if st.button("Classify Vehicle", use_container_width=True):
                with st.spinner("Analyzing..."):
                    try:
                        processed_img = preprocess_image(image)
                        predictions = model.predict(processed_img, verbose=0)
                        predicted_idx = np.argmax(predictions[0])
                        predicted_class = CLASS_NAMES[predicted_idx]
                        confidence = predictions[0][predicted_idx] * 100
                        
                        # Display only top prediction
                        st.markdown(f"**Prediction:** {predicted_class.replace('_', ' ').title()}")
                        st.markdown(f"**Confidence:** {confidence:.2f}%")
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.info("Upload an image to get started")
else:
    st.error("Model could not be loaded")

st.markdown("---")


