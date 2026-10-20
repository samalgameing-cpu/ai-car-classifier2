
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime
from pathlib import Path
import hashlib

st.set_page_config(
    page_title="AutoVision AI | Car Classifier",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Keep this order aligned with the model's original training labels.
CLASS_NAMES = [
    "Audi",
    "Hyundai Creta",
    "Mahindra Scorpio",
    "Rolls Royce",
    "Swift",
    "Tata Safari",
    "Toyota Innova"
]

MODEL_PATH = Path("final_car_model.keras")

# Premium styling and animations
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 85% 0%, #17265a 0%, transparent 38%),
        linear-gradient(135deg, #050b18, #08152b 55%, #110d28);
    color: #f4f7ff;
    font-family: 'Inter', sans-serif;
}
.block-container {
    max-width: 1500px;
    padding-top: 1.6rem;
    padding-bottom: 3rem;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071226, #0b1023);
    border-right: 1px solid #26385b;
}
.hero {
    padding: 32px;
    border: 1px solid #3c4c91;
    border-radius: 24px;
    background: linear-gradient(115deg, #102d54dd, #1a1645dd, #311345cc);
    margin-bottom: 24px;
    animation: appear .65s ease-out;
    box-shadow: 0 12px 45px #03081766;
}
.hero h1 {
    font-size: clamp(2rem, 4vw, 3.4rem);
    font-weight: 800;
    letter-spacing: -1.5px;
    background: linear-gradient(90deg, #ffffff, #63c9ff, #c48bff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.eyebrow {
    color: #79caff;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-size: .75rem;
    font-weight: 700;
}
.muted { color: #b5c5e5; }
.panel {
    padding: 22px;
    border: 1px solid #293f69;
    border-radius: 19px;
    background: linear-gradient(145deg, #0c1c35ee, #0a1228ee);
    margin-bottom: 18px;
    transition: transform .25s ease, border-color .25s ease,
                box-shadow .25s ease;
    box-shadow: 0 8px 25px #02081730;
}
.panel:hover {
    transform: translateY(-3px);
    border-color: #6c75ef;
    box-shadow: 0 10px 30px #4c42a522;
}
.metric {
    padding: 18px;
    border: 1px solid #304876;
    border-radius: 16px;
    background: linear-gradient(140deg, #102747, #171637);
    transition: transform .2s ease;
}
.metric:hover { transform: translateY(-3px); }
.result {
    border: 1px solid #6374f5;
    border-radius: 18px;
    padding: 22px;
    background: linear-gradient(120deg, #103d59, #282052);
    animation: appear .5s ease-out;
}
.car-card {
    padding: 14px;
    min-height: 100px;
    text-align: center;
    border-radius: 15px;
    border: 1px solid #2e456d;
    background: linear-gradient(145deg, #10223f, #11132b);
    transition: all .25s ease;
}
.car-card:hover {
    transform: translateY(-4px) scale(1.01);
    border-color: #9b6bff;
}
div.stButton > button {
    border: 1px solid #6d6cf4;
    border-radius: 12px;
    background: linear-gradient(90deg, #087fff, #7047ff, #bd36ef);
    color: white;
    font-weight: 700;
    min-height: 44px;
    transition: all .2s ease;
}
div.stButton > button:hover {
    border-color: #bba8ff;
    box-shadow: 0 0 22px #7253ff66;
    transform: translateY(-1px);
    color: white;
}
[data-testid="stFileUploader"] {
    border: 1px dashed #6278d8;
    border-radius: 16px;
    padding: 12px;
    background: #0c1a32;
}
hr { border-color: #293c5d; }
@keyframes appear {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: .01ms !important;
        transition-duration: .01ms !important;
    }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "final_car_model.keras is missing from the app folder."
        )
    return tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )


def predict_car(image):
    model = load_model()
    resized = image.resize((224, 224))
    array = np.asarray(resized, dtype=np.float32)
    array = np.expand_dims(array, axis=0)

    scores = np.asarray(
        model.predict(array, verbose=0)[0]
    ).flatten()

    if len(scores) != len(CLASS_NAMES):
        raise ValueError(
            f"Model returned {len(scores)} scores, "
            f"but {len(CLASS_NAMES)} class names are configured."
        )

    index = int(np.argmax(scores))
    return index, scores


# Session-only prediction history
if "history" not in st.session_state:
    st.session_state.history = []

if "last_upload_hash" not in st.session_state:
    st.session_state.last_upload_hash = None

if "current_result" not in st.session_state:
    st.session_state.current_result = None


# Sidebar
with st.sidebar:
    st.markdown("# 🚘 AutoVision AI")
    st.caption("SMART CAR RECOGNITION")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["🏠 Home", "🖼️ Car Gallery", "🕘 Prediction History",
         "🧠 Model Information", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("#### 🚗 Supported categories")
    for name in CLASS_NAMES:
        st.caption(f"✦  {name}")

    st.divider()
    st.caption("Powered by Python · TensorFlow · Streamlit")


# Hero header
st.markdown("""
<div class="hero">
  <div class="eyebrow">✦ AI POWERED · COMPUTER VISION</div>
  <h1>AI Car Classifier</h1>
  <p class="muted">
    Upload a car photo and discover its predicted category
    with your trained deep-learning model.
  </p>
</div>
""", unsafe_allow_html=True)


if page == "🏠 Home":
    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown(
            '<div class="metric">⚡ <b>AI Prediction</b>'
            '<p class="muted">Model-powered recognition</p></div>',
            unsafe_allow_html=True
        )
    with m2:
        st.markdown(
            '<div class="metric">🚘 <b>7 Categories</b>'
            '<p class="muted">Seven trained car labels</p></div>',
            unsafe_allow_html=True
        )
    with m3:
        st.markdown(
            '<div class="metric">📊 <b>Score Breakdown</b>'
            '<p class="muted">Compare all class scores</p></div>',
            unsafe_allow_html=True
        )

    left, right = st.columns([1.05, .95], gap="large")

    with left:
        st.markdown("### 📤 Upload your car")
        st.markdown(
            '<p class="muted">Use a clear JPG, JPEG or PNG photo.</p>',
            unsafe_allow_html=True
        )

        upload = st.file_uploader(
            "Choose a car image",
            type=["jpg", "jpeg", "png"],
            key="car_upload"
        )

        if upload is not None:
            try:
                image = Image.open(upload).convert("RGB")
                upload_hash = hashlib.sha256(
                    upload.getvalue()
                ).hexdigest()

                st.image(
                    image,
                    caption="Uploaded vehicle",
                    use_container_width=True
                )

                if upload_hash != st.session_state.last_upload_hash:
                    st.session_state.current_result = None

                if st.button(
                    "✨ Identify This Car",
                    type="primary",
                    use_container_width=True
                ):
                    with st.spinner("Analysing your image..."):
                        index, scores = predict_car(image)

                        result = {
                            "name": CLASS_NAMES[index],
                            "scores": scores.tolist(),
                            "confidence": float(scores[index]) * 100,
                            "time": datetime.now().strftime(
                                "%d %b %Y, %I:%M:%S %p"
                            ),
                            "image": image.copy(),
                            "image_hash": upload_hash
                        }

                        st.session_state.current_result = result
                        st.session_state.last_upload_hash = upload_hash

                        # Avoid duplicate history entries for the same
                        # image if the user repeats the same prediction.
                        history = st.session_state.history
                        if not history or history[0]["image_hash"] != upload_hash:
                            history.insert(0, result)
                            st.session_state.history = history[:20]

                        st.balloons()

            except Exception as e:
                st.error(f"Prediction failed: {e}")

        st.markdown("---")
        st.markdown("#### ✨ What can you upload?")
        cols = st.columns(3)
        tips = [
            ("📷", "Clear photo"),
            ("💡", "Good lighting"),
            ("🚘", "Visible vehicle")
        ]
        for col, (emoji, label) in zip(cols, tips):
            with col:
                st.markdown(
                    f'<div class="car-card">{emoji}<br><b>{label}</b></div>',
                    unsafe_allow_html=True
                )

    with right:
        st.markdown("### 🔮 Prediction result")
        result = st.session_state.current_result

        if result:
            st.markdown(
                f'<div class="result"><div class="eyebrow">'
                f'PREDICTED CATEGORY</div><h2>🚘 {result["name"]}</h2>'
                f'<p>Model score: <b>{result["confidence"]:.2f}%</b></p>'
                f'<p class="muted">Analysed at {result["time"]}</p></div>',
                unsafe_allow_html=True
            )

            st.progress(
                min(100, max(0, int(result["confidence"]))),
                text="Top class score"
            )

            st.markdown("#### 📊 All category scores")
            scores = result["scores"]
            for i in np.argsort(scores)[::-1]:
                score = float(scores[int(i)])
                st.write(f"**{CLASS_NAMES[int(i)]}** · {score * 100:.2f}%")
                st.progress(min(100, max(0, int(score * 100))))

            st.caption(
                "Scores are model outputs, not guaranteed real-world "
                "probabilities. Results may be incorrect."
            )
        else:
            st.markdown(
                '<div class="panel" style="text-align:center;padding:48px 12px">'
                '<div style="font-size:3rem">🚘</div>'
                '<h3>Ready to identify</h3>'
                '<p class="muted">Upload a photo and click Identify This Car.</p>'
                '</div>',
                unsafe_allow_html=True
            )

        st.markdown("#### 🕘 Recent predictions")
        if st.session_state.history:
            for item in st.session_state.history[:3]:
                st.write(
                    f"🚗 **{item['name']}** · "
                    f"{item['confidence']:.1f}% · {item['time']}"
                )
        else:
            st.caption("Your recent predictions will appear here.")


elif page == "🖼️ Car Gallery":
    st.markdown("### Explore the seven supported classes")
    st.markdown(
        '<p class="muted">Select a category to learn what the model supports. '
        'These cards describe labels; they are not prediction examples.</p>',
        unsafe_allow_html=True
    )

    descriptions = {
        "Audi": "Audi vehicle category",
        "Hyundai Creta": "Hyundai Creta SUV category",
        "Mahindra Scorpio": "Mahindra Scorpio SUV category",
        "Rolls Royce": "Rolls Royce vehicle category",
        "Swift": "Maruti Suzuki Swift hatchback category",
        "Tata Safari": "Tata Safari SUV category",
        "Toyota Innova": "Toyota Innova MPV category"
    }

    cols = st.columns(3)
    for i, name in enumerate(CLASS_NAMES):
        with cols[i % 3]:
            st.markdown(
                f'<div class="panel"><div style="font-size:2rem">🚘</div>'
                f'<h3>{name}</h3><p class="muted">'
                f'{descriptions[name]}</p></div>',
                unsafe_allow_html=True
            )

    st.info(
        "To display actual car photos in this gallery, add your own "
        "sample images to a gallery folder in the GitHub repository."
    )


elif page == "🕘 Prediction History":
    st.markdown("### 🕘 Your prediction history")
    st.caption("History is kept for this active session only.")

    if st.session_state.history:
        if st.button("🗑️ Clear history"):
            st.session_state.history = []
            st.session_state.current_result = None
            st.rerun()

        for i, item in enumerate(st.session_state.history):
            with st.expander(
                f"{item['name']} · {item['confidence']:.2f}% · {item['time']}"
            ):
                st.image(item["image"], width=320)
                st.write(f"Predicted category: **{item['name']}**")
                st.write(f"Model score: **{item['confidence']:.2f}%**")
    else:
        st.info("No predictions yet. Go to Home and analyse a car image.")


elif page == "🧠 Model Information":
    st.markdown("### 🧠 Model details")
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.write("**Architecture:** MobileNetV2-based classifier")
    st.write("**Framework:** TensorFlow / Keras")
    st.write("**Input size:** 224 × 224 RGB")
    st.write("**Output categories:** 7")
    st.write("**Model file:** `final_car_model.keras`")
    st.write("**Preprocessing:** Must match the saved model's training pipeline.")
    st.caption(
        "Accuracy is not shown here because it should be reported from "
        "the actual held-out test evaluation, not guessed."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("#### Categories")
    for name in CLASS_NAMES:
        st.write(f"• {name}")


elif page == "ℹ️ About":
    st.markdown("""
    <div class="panel">
      <h2>About AutoVision AI</h2>
      <p class="muted">
        An educational image-classification project that uses a trained
        deep-learning model to predict one of seven car categories.
      </p>
      <p class="muted">
        Predictions depend on image quality and the model's training data.
        The result should not be treated as a guaranteed identification.
      </p>
    </div>
    """, unsafe_allow_html=True)


st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#93a8cc">'
    '🚘 AutoVision AI · Built with Python, TensorFlow & Streamlit</p>',
    unsafe_allow_html=True
)
