
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="AI Car Classifier",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

CLASS_NAMES = [
    "Audi",
    "Hyundai Creta",
    "Mahindra Scorpio",
    "Rolls Royce",
    "Swift",
    "Tata Safari",
    "Toyota Innova"
]

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #06101f, #0b1530 65%, #160d2c);
    color: #f1f5ff;
}
[data-testid="stSidebar"] {
    background: #081326;
    border-right: 1px solid #243957;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}
.hero {
    padding: 28px;
    border: 1px solid #263e69;
    border-radius: 22px;
    background: linear-gradient(115deg, #10264a, #17183d, #291345);
    margin-bottom: 25px;
}
.hero h1 {
    font-size: clamp(2rem, 4vw, 3.3rem);
    margin-bottom: 8px;
    color: #ffffff;
}
.hero p { color: #c5d7f6; font-size: 1.08rem; }
.panel {
    padding: 22px;
    border: 1px solid #2a4165;
    border-radius: 18px;
    background: rgba(9, 23, 44, 0.88);
    margin-bottom: 18px;
}
.eyebrow {
    color: #8abaff;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: .75rem;
    font-weight: 700;
}
.result {
    padding: 22px;
    border-radius: 16px;
    background: linear-gradient(120deg, #123d59, #292052);
    border: 1px solid #5366d7;
}
.small-muted { color: #afc2e1; }
div.stButton > button {
    background: linear-gradient(90deg, #168bff, #9147ff);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.65rem 1rem;
    font-weight: 700;
    width: 100%;
}
div.stButton > button:hover {
    border: 1px solid #b9a5ff;
    color: white;
}
[data-testid="stFileUploader"] {
    border: 1px dashed #5176ac;
    border-radius: 15px;
    padding: 12px;
    background: #0b1930;
}
hr { border-color: #253958; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "final_car_model.keras",
        compile=False
    )

with st.sidebar:
    st.markdown("# 🚘 AutoVision AI")
    st.caption("UPLOAD  •  PREDICT  •  EXPLORE")
    st.divider()
    page = st.radio(
        "NAVIGATION",
        ["🏠 Home", "📘 About", "⚙️ How It Works", "🧠 Model Info"]
    )
    st.divider()
    st.markdown("### ✨ Smarter AI")
    st.caption("Deep learning meets automotive recognition.")
    st.caption("Built with Streamlit and TensorFlow.")

st.markdown("""
<div class="hero">
  <div class="eyebrow">AI POWERED · COMPUTER VISION</div>
  <h1>🚗 AI Car Classifier</h1>
  <p>Upload a car photo. Let artificial intelligence predict its category.</p>
</div>
""", unsafe_allow_html=True)

if page == "🏠 Home":
    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="panel">⚡ <b>Fast predictions</b><br><span class="small-muted">Deep learning model</span></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel">🎯 <b>7 car classes</b><br><span class="small-muted">Trained categories</span></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="panel">🖼️ <b>Easy image upload</b><br><span class="small-muted">JPG, JPEG and PNG</span></div>', unsafe_allow_html=True)

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown('<div class="panel"><h3>📤 Upload your car image</h3><p class="small-muted">Choose a clear photo showing the vehicle.</p></div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG"
        )

        if uploaded_file:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Your uploaded image", use_container_width=True)

                if st.button("✨ Identify This Car", type="primary"):
                    with st.spinner("AI is analysing your image..."):
                        model = load_model()
                        resized = image.resize((224, 224))
                        array = np.asarray(resized, dtype=np.float32)
                        array = np.expand_dims(array, axis=0)

                        predictions = np.asarray(
                            model.predict(array, verbose=0)[0]
                        ).flatten()

                        if len(predictions) != len(CLASS_NAMES):
                            st.error(
                                "Model output does not match the 7 class names. "
                                "Please check the model and class order."
                            )
                        else:
                            index = int(np.argmax(predictions))
                            confidence = float(predictions[index]) * 100

                            st.session_state["car_result"] = {
                                "name": CLASS_NAMES[index],
                                "confidence": confidence,
                                "scores": predictions.tolist()
                            }
            except Exception as e:
                st.error(f"Could not process the image: {e}")

    with right:
        st.markdown('<div class="panel"><h3>🔮 Prediction Result</h3><p class="small-muted">Your AI result will appear here.</p></div>', unsafe_allow_html=True)

        result = st.session_state.get("car_result")

        if result:
            st.markdown(
                f'<div class="result"><div class="eyebrow">PREDICTED CATEGORY</div>'
                f'<h2>🚘 {result["name"]}</h2>'
                f'<p>Model confidence: <b>{result["confidence"]:.2f}%</b></p></div>',
                unsafe_allow_html=True
            )
            st.progress(
                min(100, max(0, int(result["confidence"]))),
                text="Prediction confidence"
            )

            st.markdown("#### 📊 Class scores")
            scores = result["scores"]
            for i in np.argsort(scores)[::-1]:
                st.write(
                    f"**{CLASS_NAMES[int(i)]}** — "
                    f"{float(scores[int(i)]) * 100:.2f}%"
                )
        else:
            st.info("Upload an image and click Identify This Car to see your result.")
            st.markdown("#### 🚙 Available categories")
            for name in CLASS_NAMES:
                st.write(f"• {name}")

    st.divider()
    st.markdown("### 💡 Tips for better predictions")
    t1, t2, t3 = st.columns(3)
    t1.info("Use a clear, well-lit car photo.")
    t2.info("Keep the vehicle visible in the frame.")
    t3.info("Try a different angle if the result looks incorrect.")

elif page == "📘 About":
    st.markdown('<div class="panel"><h2>About this project</h2><p>This project uses a trained deep-learning image classification model to predict which of seven car categories an uploaded image belongs to.</p></div>', unsafe_allow_html=True)
    st.markdown("#### Supported categories")
    for name in CLASS_NAMES:
        st.write(f"🚗 {name}")

elif page == "⚙️ How It Works":
    st.markdown("### From image to prediction")
    for n, title, desc in [
        (1, "Upload", "Select a JPG, JPEG or PNG car image."),
        (2, "Preprocess", "The image is resized to 224 × 224 pixels."),
        (3, "Predict", "The trained model calculates class scores."),
        (4, "Display", "The app shows the highest-scoring class and confidence.")
    ]:
        st.markdown(
            f'<div class="panel"><h3>{n}. {title}</h3><p class="small-muted">{desc}</p></div>',
            unsafe_allow_html=True
        )

elif page == "🧠 Model Info":
    st.markdown("### Model information")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.write("**Architecture:** MobileNetV2-based image classifier")
    st.write("**Framework:** TensorFlow / Keras")
    st.write("**Input size:** 224 × 224 RGB")
    st.write("**Output classes:** 7")
    st.write("**Model file:** `final_car_model.keras`")
    st.caption("Actual accuracy depends on the saved model and evaluation dataset. No unverified accuracy figure is displayed.")
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.caption("AI Car Classifier · Educational project · Predictions may be incorrect.")
