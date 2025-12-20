import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# App title
st.title("🚗 AI Vehicle Image Classification")

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5
CLASSES = ["Bike", "Bus", "Car"]

# Train the model (cached so it doesn't retrain every interaction)
@st.cache_resource
def train_model():
    # Image preprocessing
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Load datasets
    train_data = train_datagen.flow_from_directory(
        "dataset/train",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    test_data = test_datagen.flow_from_directory(
        "dataset/test",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    # Build CNN model
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(len(CLASSES), activation="softmax")
    ])

    # Compile model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Train model
    st.info("Training model, please wait...")
    model.fit(train_data, epochs=EPOCHS, validation_data=test_data)
    st.success("Model training completed!")
    return model

# Load or train model
model = train_model()

# Upload image
uploaded_file = st.file_uploader("Upload a vehicle image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_array = np.array(image)/255.0
    img_array = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 3)

    # Predict
    prediction = model.predict(img_array)
    st.success(f"Predicted Vehicle: {CLASSES[np.argmax(prediction)]}")

