import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import requests
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Emotion Recognition App",
    page_icon="😊",
    layout="wide"
)

# --- Custom CSS for Styling ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# You would need to create a style.css file for this to work
# As a fallback, I will embed the CSS directly.
st.markdown("""
<style>
    /* General Styling */
    body {
        background-color: #1a1a2e;
        color: #e0e0e0;
    }
    .main {
        background: linear-gradient(135deg, #16222A 0%, #3A6073 100%);
        padding: 2rem;
        border-radius: 15px;
    }
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 20px;
        padding: 10px 20px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stFileUploader, .stSelectbox, .stNumberInput {
        border-radius: 10px;
    }
    /* Custom Components */
    .emotion-badge {
        display: inline-block;
        padding: 0.5em 0.9em;
        font-size: 0.9em;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.75rem;
        color: #fff;
        margin-right: 5px;
    }
    footer {
        text-align: center;
        padding: 1rem;
        color: #a0a0a0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# --- Model and Classifier Loading ---
# NEW MODEL: This is a publicly available model compatible with modern TensorFlow.
MODEL_URL = "https://github.com/drishtimittal/Facial-Emotion-Recognition/raw/main/model.h5"
MODEL_PATH = "emotion_model.h5"
FACE_CLASSIFIER_PATH = 'haarcascade_frontalface_default.xml'

@st.cache_data
def download_file(url, file_path):
    """Downloads a file from a URL and saves it locally."""
    if not os.path.exists(file_path):
        with st.spinner(f"Downloading {os.path.basename(file_path)}... This may take a moment."):
            try:
                r = requests.get(url, stream=True)
                r.raise_for_status()
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success(f"✅ {os.path.basename(file_path)} downloaded successfully!")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Error downloading file: {e}")
                return False
    return True

@st.cache_resource
def load_emotion_model():
    """Loads the emotion detection model from the local file."""
    if not download_file(MODEL_URL, MODEL_PATH):
        return None
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        st.success("✅ Emotion model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model file: {e}")
        st.info("The model file might be incompatible. Please ensure it was trained with a compatible TensorFlow version.")
        return None

@st.cache_resource
def load_face_classifier():
    """Loads the Haar Cascade face classifier."""
    if not os.path.exists(FACE_CLASSIFIER_PATH):
        st.error(f"Face classifier file '{FACE_CLASSIFIER_PATH}' not found.")
        return None
    return cv2.CascadeClassifier(FACE_CLASSIFIER_PATH)

# --- Core Emotion Detection Logic ---
# UPDATED for the new model's requirements (48x48 grayscale)
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
EMOTION_COLORS = {
    'Angry': '#d32f2f', 'Disgust': '#7b1fa2', 'Fear': '#f57c00', 'Happy': '#388e3c',
    'Neutral': '#757575', 'Sad': '#1976d2', 'Surprise': '#fbc02d'
}

def detect_emotions(image, model, face_classifier):
    """Detects faces and predicts emotions for each face found in an image."""
    if isinstance(image, Image.Image):
        image = np.array(image)

    if image.dtype != np.uint8:
        image = (255 * (image / np.max(image))).astype(np.uint8)

    if len(image.shape) > 2 and image.shape[2] == 4: # Handle RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    color_image = image.copy()

    faces = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    results = []
    for (x, y, w, h) in faces:
        # Preprocess the face for the new model (48x48 grayscale)
        face_roi = gray_image[y:y+h, x:x+w]
        resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
        normalized_face = resized_face / 255.0
        reshaped_face = np.reshape(normalized_face, (1, 48, 48, 1))

        # Predict emotion
        prediction = model.predict(reshaped_face, verbose=0)[0]
        emotion_index = np.argmax(prediction)
        emotion = EMOTION_LABELS[emotion_index]
        confidence = prediction[emotion_index]

        # Draw on image
        color = tuple(int(EMOTION_COLORS[emotion].lstrip('#')[i:i+2], 16) for i in (4, 2, 0)) # BGR for OpenCV
        cv2.rectangle(color_image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(color_image, f"{emotion} ({confidence:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        results.append({
            'emotion': emotion,
            'confidence': confidence,
            'bbox': (x, y, w, h),
            'all_predictions': dict(zip(EMOTION_LABELS, prediction))
        })
    return color_image, results

# --- UI Components ---
def display_results(image, results):
    """Displays the processed image and detailed results in tabs."""
    if not results:
        st.warning("No faces were detected in the image.")
        return

    tab_titles = ["Summary"] + [f"Face {i+1}" for i in range(len(results))]
    tabs = st.tabs(tab_titles)

    with tabs[0]:
        st.subheader("Emotion Summary")
        emotion_counts = pd.Series([res['emotion'] for res in results]).value_counts()
        st.bar_chart(emotion_counts)

    for i, result in enumerate(results):
        with tabs[i+1]:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader(f"Face {i+1}")
                badge_color = EMOTION_COLORS.get(result['emotion'], '#808080')
                st.markdown(f'<span class="emotion-badge" style="background-color:{badge_color};">{result["emotion"]}</span>', unsafe_allow_html=True)
                st.write(f"**Confidence:** {result['confidence']:.2%}")

            with col2:
                st.subheader("Emotion Probabilities")
                prob_df = pd.DataFrame.from_dict(result['all_predictions'], orient='index', columns=['Probability'])
                prob_df = prob_df.sort_values(by='Probability', ascending=False)
                st.dataframe(prob_df.style.format("{:.2%}"))

def image_to_bytes(image):
    """Converts a NumPy image to bytes for downloading."""
    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return buffer.tobytes()

# --- Main App Logic ---
def main():
    st.title("🎭 Advanced Emotion Recognition")
    st.write("Upload an image and the AI will detect the emotions in each face.")

    model = load_emotion_model()
    face_classifier = load_face_classifier()

    if model is None or face_classifier is None:
        st.error("🔴 Critical components failed to load. The app cannot proceed. Please check the logs.")
        return

    st.sidebar.title("🖼️ Input Options")
    input_method = st.sidebar.radio("Choose how to provide an image:", ["Upload an Image", "Take a Photo"])

    image_file = None
    if input_method == "Upload an Image":
        image_file = st.sidebar.file_uploader("Select an image", type=["jpg", "jpeg", "png"])
    elif input_method == "Take a Photo":
        image_file = st.sidebar.camera_input("Smile for the camera!")

    if image_file:
        original_image = Image.open(image_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(original_image, use_container_width=True)

        with st.spinner("Analyzing emotions..."):
            processed_image, results = detect_emotions(original_image, model, face_classifier)

        with col2:
            st.subheader("Processed Image")
            st.image(processed_image, use_container_width=True)
            st.download_button(
                label="📥 Download Processed Image",
                data=image_to_bytes(processed_image),
                file_name="emotion_analysis.png",
                mime="image/png"
            )
        
        st.divider()
        display_results(processed_image, results)

    st.markdown("---")
    st.markdown('<footer>Built with ❤️ using Streamlit and TensorFlow.</footer>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

