
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI Car Classifier", page_icon="🚗")

CLASS_NAMES = [
    "Audi",
    "Hyundai Creta",
    "Mahindra Scorpio",
    "Rolls Royce",
    "Swift",
    "Tata Safari",
    "Toyota Innova"
]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "final_car_model.keras",
        compile=False
    )

st.title("🚗 AI Car Classifier")
st.write("Upload a car image to identify its category.")

uploaded_file = st.file_uploader(
    "Choose a car image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded car", use_container_width=True)

    if st.button("Identify Car"):
        try:
            model = load_model()
            image = image.resize((224, 224))
            image_array = np.array(image, dtype=np.float32)
            image_array = np.expand_dims(image_array, axis=0)

            predictions = model.predict(image_array, verbose=0)[0]
            index = int(np.argmax(predictions))

            st.success(f"Predicted Car: {CLASS_NAMES[index]}")
            st.write(
                f"Model confidence: {predictions[index] * 100:.2f}%"
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")
