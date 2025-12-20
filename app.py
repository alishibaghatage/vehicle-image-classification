import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.title("🚗 AI Vehicle Image Classification")

IMG_SIZE = 224

# Load trained model (trained locally)
@st.cache_resource
def load_trained_model():
    return load_model("vehicle_model.h5")

model = load_trained_model()

classes = ["Bike", "Bus", "Car"]

uploaded_file = st.file_uploader(
    "Upload a vehicle image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 3)

    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]

    st.success(f"Predicted Vehicle: **{predicted_class}**")
