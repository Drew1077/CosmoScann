import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('model.h5')

# UI Title
st.title("🧠 Retina-based Alzheimer's Risk Predictor")

# File uploader
uploaded_file = st.file_uploader("📤 Upload a Retina Image", type=["jpg", "jpeg", "png"])

# Process after upload
if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Retina", use_column_width=True)

    # Preprocess the image
    img = img.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    label = 1 if prediction >= 0.5 else 0

    # Show result
    st.markdown(f"### 📊 Prediction: **{label}**")
    st.markdown("*(1 = Potential Risk of Alzheimer's, 0 = Low Risk)*")


