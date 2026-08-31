import json
import numpy as np
from PIL import Image
import streamlit as st

st.title("Plant Disease Detector")
st.write("Upload a leaf photo to diagnose diseases.")


# Load JSON class mapping
@st.cache_resource
def load_files():
  with open("class_indices.json", "r") as f:
    indices = json.load(f)
  labels = {int(v): k for k, v in indices.items()}
  return labels


labels = load_files()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
  image = Image.open(uploaded_file)
  st.image(image, caption="Uploaded Image", use_column_width=True)

  if st.button("Predict"):
    # Simulated prediction output
    class_id = 0
    confidence = 95.50

    predicted_label = labels.get(class_id, "Healthy")
    st.success(f"Result: {predicted_label}")
    st.info(f"Confidence: {confidence:.2f}%")

