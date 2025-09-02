import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import io
import requests  # Import the requests library
from contextlib import contextmanager

# --- Page Configuration ---
st.set_page_config(
    page_title="Emotion Detection App",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Styling ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# You would create a style.css file for this, but for a single file we inject it directly
st.markdown("""
<style>
    /* General Styles */
    body {
        color: #E0E0E0;
        background-color: #1E1E1E;
    }
    .stApp {
        background: linear-gradient(135deg, #232526 0%, #414345 100%);
    }
    .st-emotion-cache-16txtl3 {
        padding: 2rem 1rem 1rem;
    }

    /* Titles and Headers */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        color: #FFFFFF;
    }
    h1 {
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    /* Sidebar Styles */
    .st-emotion-cache-10oheav {
        background-color: rgba(40, 40, 40, 0.8);
        border-right: 1px solid #444;
    }
    .st-emotion-cache-10oheav .stSelectbox, .st-emotion-cache-10oheav .stSlider {
        margin-bottom: 1rem;
    }

    /* Buttons and Widgets */
    .stButton>button {
        background-color: #007BFF;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: background-color 0.3s ease, transform 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        transform: scale(1.05);
    }
    .stFileUploader, .stCameraInput {
        border: 2px dashed #555;
        border-radius: 12px;
        padding: 1.5rem;
        background-color: rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)


# --- TensorFlow Import with Error Handling ---
@contextmanager
def suppress_tf_warnings():
    original_level = os.environ.get('TF_CPP_MIN_LOG_LEVEL', '0')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING messages
    try:
        yield
    finally:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = original_level

try:
    with suppress_tf_warnings():
        import tensorflow as tf
        from tensorflow.keras.preprocessing.image import img_to_array
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    st.error("❌ TensorFlow not found. Please ensure it's installed.")

# --- Model and Classifier Loading ---
MODEL_URL = "https://www.dropbox.com/scl/fi/9nly99wm1e8405sknc3ho/emotion_model.h5?rlkey=zrw75xjq6xkuvwqmcd9nwmr5l&st=ur3a6le5&dl=1" # IMPORTANT: Update this link
MODEL_PATH = "emotion_model.h5"

@st.cache_resource
def download_file(url, local_filename):
    """Downloads a file from a URL and saves it locally."""
    if not os.path.exists(local_filename):
        with st.spinner(f"Downloading model... (this may take a moment)"):
            try:
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(local_filename, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                st.success("✅ Model downloaded successfully!")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Error downloading model: {e}")
                return False
    return True

@st.cache_resource
def load_model():
    """Loads the pre-trained emotion detection model."""
    if not TF_AVAILABLE:
        st.warning("⚠️ TensorFlow not available - cannot load model.")
        return None

    if not download_file(MODEL_URL, MODEL_PATH):
        return None

    if os.path.exists(MODEL_PATH):
        try:
            with suppress_tf_warnings():
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            return model
        except Exception as e:
            st.error(f"❌ Error loading model file: {e}")
            st.info("The model file might be corrupted. Please try deleting it and reloading the app to re-download.")
            return None
    else:
        st.warning("⚠️ Model file not found even after download attempt.")
        return None

@st.cache_resource
def load_face_classifier():
    """Loads the Haar Cascade face classifier."""
    classifier_path = 'haarcascade_frontalface_default.xml'
    if not os.path.exists(classifier_path):
        st.error(f"Face classifier file '{classifier_path}' not found.")
        return None
    return cv2.CascadeClassifier(classifier_path)

# --- Core Emotion Detection Logic ---
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
emotion_colors = {
    'Angry': (255, 0, 0), 'Disgust': (0, 128, 0), 'Fear': (128, 0, 128),
    'Happy': (255, 255, 0), 'Neutral': (128, 128, 128), 'Sad': (0, 0, 255),
    'Surprise': (255, 165, 0)
}

def detect_emotion(image, model, face_classifier, scale_factor, min_neighbors):
    """Detects faces and predicts emotions in an image."""
    if model is None or face_classifier is None:
        return image, []

    if isinstance(image, Image.Image):
        image = np.array(image.convert('RGB'))

    if image.dtype != np.uint8:
        image = (255 * (image / np.max(image))).astype(np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    colored_image = image.copy()

    faces = face_classifier.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(48, 48)
    )

    results = []
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48))
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_RGB2GRAY)

        face_processed = face_gray.astype("float32") / 255.0
        face_processed = img_to_array(face_processed)
        face_processed = np.expand_dims(face_processed, axis=0)

        with suppress_tf_warnings():
            prediction = model.predict(face_processed, verbose=0)[0]
        
        emotion_idx = np.argmax(prediction)
        emotion = emotion_labels[emotion_idx]
        confidence = prediction[emotion_idx]
        color = emotion_colors.get(emotion, (0, 255, 0))

        # Draw rectangle and text
        cv2.rectangle(colored_image, (x, y), (x+w, y+h), color, 2)
        label_text = f"{emotion}: {confidence:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(colored_image, (x, y - text_h - 10), (x + text_w + 4, y - 5), color, -1)
        cv2.putText(colored_image, label_text, (x + 2, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        results.append({
            'emotion': emotion,
            'confidence': confidence,
            'bbox': (x, y, w, h),
            'all_predictions': dict(zip(emotion_labels, prediction))
        })

    return colored_image, results


# --- UI and Main Application ---
def main():
    st.title("🎭 Advanced Emotion Recognition")

    # Load resources
    model = load_model()
    face_classifier = load_face_classifier()

    if model is None or face_classifier is None:
        st.error("🔴 Critical components failed to load. The app cannot proceed. Please check the logs.")
        st.stop()

    # --- Sidebar for Options ---
    with st.sidebar:
        st.header("⚙️ Options")
        input_method = st.radio("Choose input method:", ["Upload Image", "Take Photo"])
        
        st.subheader("🛠️ Detection Parameters")
        scale_factor = st.slider("Scale Factor", 1.05, 1.4, 1.1, 0.05, help="How much the image size is reduced at each image scale.")
        min_neighbors = st.slider("Minimum Neighbors", 3, 10, 5, 1, help="How many neighbors each candidate rectangle should have to retain it.")
        
        st.subheader("ℹ️ About")
        st.info("This app uses a deep learning model to detect emotions from faces in real-time. Upload an image or use your webcam!")

    # --- Main Panel for Input and Output ---
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            process_and_display_image(uploaded_file, model, face_classifier, scale_factor, min_neighbors)

    elif input_method == "Take Photo":
        picture = st.camera_input("Take a picture")
        if picture:
            process_and_display_image(picture, model, face_classifier, scale_factor, min_neighbors)

def process_and_display_image(image_file, model, face_classifier, scale_factor, min_neighbors):
    """Handles image processing and displays results."""
    image = Image.open(image_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with st.spinner("Analyzing emotions..."):
        processed_image, results = detect_emotion(np.array(image), model, face_classifier, scale_factor, min_neighbors)

    with col2:
        st.subheader("Detection Results")
        st.image(processed_image, use_container_width=True)

    if not results:
        st.warning("No faces detected in the image.")
    else:
        st.success(f"Detected {len(results)} face(s).")
        
        # Display results in tabs for each face
        tab_titles = [f"Face {i+1}" for i in range(len(results))]
        tabs = st.tabs(tab_titles)

        for i, (tab, result) in enumerate(zip(tabs, results)):
            with tab:
                st.write(f"**Primary Emotion:** {result['emotion']} (Confidence: {result['confidence']:.2f})")
                
                # Display all probabilities as a bar chart
                import pandas as pd
                df = pd.DataFrame(result['all_predictions'].values(), index=result['all_predictions'].keys(), columns=['Probability'])
                st.bar_chart(df)

if __name__ == "__main__":
    main()

