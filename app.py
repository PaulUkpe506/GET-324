import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.title("Plant Disease Detector")
st.write("Upload a leaf photo to diagnose diseases.")


@st.cache_resource
def load_files():
  model = tf.keras.models.load_model("best_plant_disease_model.h5")
  with open("class_indices.json", "r") as f:
    indices = json.load(f)
  labels = {v: k for k, v in indices.items()}
  return model, labels


model, labels = load_files()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
  image = Image.open(uploaded_file)
  st.image(image, caption="Uploaded Image", use_column_width=True)

  if st.button("Predict"):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    class_id = np.argmax(preds[0])
    confidence = np.max(preds[0]) * 100

    st.success(f"Result: {labels[class_id]}")
    st.info(f"Confidence: {confidence:.2f}%")
