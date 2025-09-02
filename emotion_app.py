import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import io
import pandas as pd
from collections import Counter

# ✅ Configure Streamlit page (must be first Streamlit call)
st.set_page_config(
    page_title="Emotion Detection App",
    page_icon="😊",
    layout="wide"
)

# --- Color and Style Definitions ---
EMOTION_COLORS = {
    "Happy": "#4CAF50",      # Green
    "Sad": "#2196F3",        # Blue
    "Angry": "#F44336",      # Red
    "Surprise": "#FFC107",   # Amber
    "Neutral": "#9E9E9E",    # Grey
    "Fear": "#673AB7",       # Deep Purple
    "Disgust": "#795548"     # Brown
}

def load_css():
    """Injects custom CSS for styling the app."""
    st.markdown("""
        <style>
            /* Main app background */
            .stApp {
                background-image: linear-gradient(to top right, #0a071d, #1f1a3e, #3a2f6b);
                background-attachment: fixed;
                color: #e0e0e0;
            }
            /* Sidebar styling */
            [data-testid="stSidebar"] {
                background-color: rgba(10, 7, 29, 0.8);
                border-right: 1px solid #4a3f8a;
            }
            /* Headers and titles */
            h1, h2, h3 {
                color: #a79aff; /* A nice lavender color for headers */
            }
            /* Custom emotion badges */
            .emotion-badge {
                padding: 5px 15px;
                border-radius: 20px;
                color: white;
                font-weight: bold;
                display: inline-block;
                margin-bottom: 10px;
                font-size: 1.1em;
            }
            /* Footer styling */
            .footer {
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background-color: rgba(10, 7, 29, 0.8);
                color: #a79aff;
                text-align: center;
                padding: 10px;
                font-size: 0.9em;
            }
        </style>
    """, unsafe_allow_html=True)

# --- TensorFlow Import and Setup ---
TF_AVAILABLE = False
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import img_to_array
    TF_AVAILABLE = True
except ImportError:
    pass # Errors will be handled gracefully in main app logic

# --- Model Loading ---
@st.cache_resource
def load_model():
    if not TF_AVAILABLE: return None
    model_paths = ['emotion_model.h5', 'Custom_CNN_model.keras']
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                return model
            except Exception:
                continue
    return None

def create_fallback_model():
    if not TF_AVAILABLE: return None
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2, 2), tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dense(7, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Face Classifier Loading ---
@st.cache_resource
def load_face_classifier():
    classifier_path = 'haarcascade_frontalface_default.xml'
    if not os.path.exists(classifier_path): return None
    return cv2.CascadeClassifier(classifier_path)

# --- Emotion Detection Logic ---
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def detect_emotion(image, model, face_classifier, scale_factor, min_neighbors):
    if model is None or face_classifier is None: return np.array(image), []
    if isinstance(image, Image.Image): image = np.array(image.convert('RGB'))
    processed_image, gray = image.copy(), cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(40, 40))
    results = []
    for (x, y, w, h) in faces:
        cv2.rectangle(processed_image, (x, y), (x + w, y + h), (36, 255, 12), 2)
        face_roi = image[y:y + h, x:x + w]
        face_resized = cv2.resize(face_roi, (224, 224))
        face_normalized = face_resized.astype("float32") / 255.0
        face_expanded = np.expand_dims(img_to_array(face_normalized), axis=0)
        prediction = model.predict(face_expanded, verbose=0)[0]
        emotion_idx = np.argmax(prediction)
        emotion = emotion_labels[emotion_idx]
        confidence = prediction[emotion_idx]
        label = f"{emotion} ({confidence:.2f})"
        cv2.putText(processed_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        results.append({'emotion': emotion, 'confidence': confidence, 'all_predictions': dict(zip(emotion_labels, prediction))})
    return processed_image, results

# --- Main Application UI ---
def main():
    load_css()
    st.title("🎭 Emotion Detection AI")
    st.markdown("An interactive AI-powered application to analyze emotions from facial expressions.")

    if not TF_AVAILABLE:
        st.error("❌ **TensorFlow is not installed.** The app cannot run without it.")
        st.info("🔧 Please install TensorFlow in your environment: `pip install tensorflow`")
        st.stop()
        
    model = load_model()
    face_classifier = load_face_classifier()
    
    if face_classifier is None:
        st.error("❌ **`haarcascade_frontalface_default.xml` not found.** This file is required for face detection.")
        st.stop()

    is_fallback = False
    if model is None:
        model = create_fallback_model()
        is_fallback = True
        st.warning("⚠️ **No trained model found.** Using an untrained fallback model for demonstration. Predictions will be random.")
    else:
        st.success("✅ **AI Model loaded successfully.** Real emotion detection is enabled.")

    # --- Sidebar ---
    st.sidebar.title("🚀 Options")
    input_method = st.sidebar.selectbox("Choose input method:", ["Upload Image", "Take Photo", "Upload Multiple Images"])
    st.sidebar.title("⚙️ Detection Parameters")
    scale_factor = st.sidebar.slider("Scale Factor", 1.05, 1.4, 1.1, 0.05, help="Controls how much the image size is reduced at each scale. Smaller values find more faces but are slower.")
    min_neighbors = st.sidebar.slider("Minimum Neighbors", 1, 10, 5, 1, help="Controls the sensitivity of detection. Higher values result in fewer, higher-quality detections.")
    st.sidebar.info("🧠 **About the Model**\n\nThis app uses a deep learning model to recognize 7 human emotions. The face detection is performed using OpenCV's Haar Cascades.")

    # --- Main Content ---
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            process_and_display_image(image, uploaded_file.name, model, face_classifier, scale_factor, min_neighbors)
    
    elif input_method == "Take Photo":
        picture = st.camera_input("Take a single picture")
        if picture:
            image = Image.open(picture)
            process_and_display_image(image, "webcam_capture.png", model, face_classifier, scale_factor, min_neighbors)
            
    elif input_method == "Upload Multiple Images":
        handle_batch_upload(model, face_classifier, scale_factor, min_neighbors)

    # --- Footer ---
    st.markdown('<div class="footer">Developed with ❤️ using Streamlit & TensorFlow</div>', unsafe_allow_html=True)

def handle_batch_upload(model, face_classifier, scale_factor, min_neighbors):
    uploaded_files = st.file_uploader("Choose multiple images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = []

    if uploaded_files:
        st.session_state.batch_results = [] # Reset on new upload
        with st.spinner("Processing all images... This may take a moment."):
            for f in uploaded_files:
                image = Image.open(f)
                processed_image, results = detect_emotion(image, model, face_classifier, scale_factor, min_neighbors)
                st.session_state.batch_results.append({'name': f.name, 'image': image, 'processed': processed_image, 'results': results})

    if st.session_state.batch_results:
        st.header("📦 Batch Processing Results")
        
        # --- Emotion Filtering ---
        all_detected_emotions = set(res['emotion'] for item in st.session_state.batch_results for res in item['results'])
        if all_detected_emotions:
            selected_emotions = st.multiselect("Filter by Emotion:", sorted(list(all_detected_emotions)), default=sorted(list(all_detected_emotions)))
        
        # --- Display Filtered Results ---
        filtered_items = [
            item for item in st.session_state.batch_results 
            if any(res['emotion'] in selected_emotions for res in item['results'])
        ] if all_detected_emotions else st.session_state.batch_results

        for item in filtered_items:
            with st.expander(f"Results for: {item['name']}", expanded=False):
                process_and_display_image(item['image'], item['name'], model, face_classifier, scale_factor, min_neighbors, is_batch_item=True, processed_data=item)
        
        # --- Aggregated Summary ---
        st.header("📊 Aggregated Emotion Summary")
        all_emotions = [res['emotion'] for item in filtered_items for res in item['results']]
        if all_emotions:
            emotion_counts = Counter(all_emotions)
            summary_df = pd.DataFrame(emotion_counts.items(), columns=['Emotion', 'Count'])
            st.bar_chart(summary_df.set_index('Emotion'))
        else:
            st.info("No emotions detected in the filtered set.")

def process_and_display_image(image, caption, model, face_classifier, scale_factor, min_neighbors, is_batch_item=False, processed_data=None, display_summary=True):
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption=f"Original: {caption}", use_container_width=True)

    if not is_batch_item:
        with st.spinner("Detecting emotions..."):
            processed_image_np, results = detect_emotion(image, model, face_classifier, scale_factor, min_neighbors)
    else:
        processed_image_np, results = processed_data['processed'], processed_data['results']

    with col2:
        st.image(processed_image_np, caption=f"Processed: {caption}", use_container_width=True)

    if results:
        summary_tab, *face_tabs = st.tabs(["📊 Summary", *[f"Face {i+1}" for i in range(len(results))]])
        
        with summary_tab:
            st.subheader("Emotion Overview")
            for i, result in enumerate(results):
                badge_color = EMOTION_COLORS.get(result['emotion'], '#808080')
                st.markdown(f"**Face {i+1}:** <span class='emotion-badge' style='background-color:{badge_color}'>{result['emotion']}</span> (Confidence: {result['confidence']:.2f})", unsafe_allow_html=True)
        
        for i, (tab, result) in enumerate(zip(face_tabs, results)):
            with tab:
                prob_df = pd.DataFrame(result['all_predictions'].items(), columns=['Emotion', 'Probability'])
                st.bar_chart(prob_df.set_index('Emotion'))

        # --- Download Buttons ---
        buf = io.BytesIO()
        Image.fromarray(processed_image_np).save(buf, format="PNG")
        st.download_button(label="📥 Download Annotated Image", data=buf.getvalue(), file_name=f"annotated_{caption}.png", mime="image/png")
    
    elif image is not None:
        st.warning("No faces were detected in this image.")
    
    return processed_image_np, results

if __name__ == "__main__":
    main()

