import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import time # Import the time module for delays

# --- Configuration for YOLOv8 Object Detection ---
CLASS_NAMES_YOLO = ['Paper', 'Cardboard', 'Plastic', 'Metal', 'Trash', 'Glass']
MODEL_PATH_YOLO = 'best.pt'
PADDING_PIXELS = 20

# --- Configuration for EfficientNetV2S Classification ---
MODEL_PATH_CLASSIFICATION = '224MobilenetV2.keras'
CLASS_NAMES_CLASSIFICATION = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

# --- Load Models (with caching for Streamlit) ---
@st.cache_resource
def load_yolo_model(path):
    try:
        model = YOLO(path)
        st.success(f"YOLOv8 model loaded successfully from {path}")
        return model
    except Exception as e:
        st.error(f"Error loading YOLOv8 model from {path}: {e}")
        st.info("Please ensure the YOLOv8 model path is correct and the model file exists.")
        st.stop() # Stop the app if model fails to load

@st.cache_resource
def load_classification_model(path):
    try:
        model = tf.keras.models.load_model(path)
        st.success(f"Classification model loaded successfully from {path}")
        return model
    except Exception as e:
        st.error(f"Error loading classification model from {path}: {e}")
        st.info("Please ensure the Keras model path is correct and the model file exists.")
        st.stop() # Stop the app if model fails to load

yolo_model = load_yolo_model(MODEL_PATH_YOLO)
classification_model = load_classification_model(MODEL_PATH_CLASSIFICATION)

# --- Classification Function for Cropped Images ---
def classify_cropped_image(img_np: np.ndarray) -> tuple[str, str, str]:
    """
    Classifies a single cropped image using the loaded EfficientNetV2S model.

    Args:
        img_np (np.ndarray): The input cropped image as a NumPy array.

    Returns:
        tuple[str, str, str]: A tuple containing:
            - str: A formatted string with the predicted class and confidence (e.g., "Paper (Conf: 0.95)").
            - str: An HTML string for the individual prediction status (e.g., "Predicted: Cardboard (Conf: 98.76%)").
            - str: The raw predicted label (e.g., "Cardboard").
    """
    img_pil = Image.fromarray(img_np.astype('uint8'))
    img_pil = img_pil.resize((224, 224))
    img_array = np.array(img_pil, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Introduce a delay for EfficientNet classification
    time.sleep(0.05) # Reduced delay for better Streamlit responsiveness

    prediction = classification_model.predict(img_array, verbose=0)
    predicted_index = np.argmax(prediction)

    classification_result_str = ""
    individual_prediction_html = ""
    predicted_label_raw = ""

    if predicted_index < len(CLASS_NAMES_CLASSIFICATION):
        predicted_label_raw = CLASS_NAMES_CLASSIFICATION[predicted_index]
        confidence = prediction[0][predicted_index]
        classification_result_str = f"{predicted_label_raw} (Conf: {confidence:.2f})"
        
        individual_prediction_html = f"""
        <div style='text-align: center; font-size: 28px; font-weight: bold; color: #fbff00; font-family: Arial, sans-serif;'>
            Prediction: {predicted_label_raw}<br>
            <span style='font-size: 20px; color: #888;'>Confidence: {confidence:.2%}</span>
        </div>
        """
    else:
        classification_result_str = "Prediction Error"
        individual_prediction_html = f"""
        <div style='text-align: center; font-size: 20px; color: red; font-family: Arial, sans-serif;'>
            Classification Error: Class index {predicted_index} out of bounds.
        </div>
        """
    return classification_result_str, individual_prediction_html, predicted_label_raw


# --- Combined Object Detection and Classification Function ---
def detect_and_classify_objects_streamlit(image_np: np.ndarray):
    """
    Performs object detection, crops detected objects, and then classifies each cropped object.
    If no objects are detected, it classifies the entire image. This function is adapted for Streamlit.

    Args:
        image_np (np.ndarray): The input image frame as a NumPy array (H, W, C).

    Returns:
        tuple[np.ndarray, list, str]: A tuple containing:
            - np.ndarray: The image frame with bounding boxes drawn on it (or the original if no detections).
            - list: A list of (cropped_image_np, classification_label_string) tuples for the gallery.
            - str: An HTML string representing the final prediction status.
    """
    annotated_frame = image_np.copy()
    cropped_images_with_labels = []
    
    # Perform YOLOv8 inference
    results = yolo_model(image_np, conf=0.5, iou=0.45, verbose=False)

    detections_found = False
    all_boxes_data = [] # Store box data for drawing and later classification
    
    for r in results:
        if len(r.boxes) > 0:
            detections_found = True
            for box in r.boxes:
                all_boxes_data.append(box)
                x1_orig, y1_orig, x2_orig, y2_orig = map(int, box.xyxy[0])
                color = (0, 255, 0)
                cv2.rectangle(annotated_frame, (x1_orig, y1_orig), (x2_orig, y2_orig), color, 2)
    
    final_prediction_status_html = "" # This will hold the HTML for the very last prediction

    if detections_found:
        for i, box in enumerate(all_boxes_data):
            x1_orig, y1_orig, x2_orig, y2_orig = map(int, box.xyxy[0])

            h, w, _ = image_np.shape
            x1_padded = max(0, x1_orig - PADDING_PIXELS)
            y1_padded = max(0, y1_orig - PADDING_PIXELS)
            x2_padded = min(w, x2_orig + PADDING_PIXELS)
            y2_padded = min(h, y2_orig + PADDING_PIXELS)

            if x2_padded > x1_padded and y2_padded > y1_padded:
                cropped_object_np = image_np[y1_padded:y2_padded, x1_padded:x2_padded]

                classification_result_str, individual_prediction_html, predicted_label = classify_cropped_image(cropped_object_np)
                
                final_prediction_status_html = individual_prediction_html 

                gallery_image = cropped_object_np.copy()
                cropped_images_with_labels.append((gallery_image, classification_result_str))
                
    else:
        # If no objects detected by YOLO, classify the entire image
        st.info("No objects detected by YOLO. Classifying the entire image.")
        classification_result_str, individual_prediction_html, predicted_label = classify_cropped_image(image_np)
        
        final_prediction_status_html = individual_prediction_html.replace("Prediction:", "Prediction (Full Image):") # Adjust text for full image

        gallery_image = image_np.copy()
        cropped_images_with_labels.append((gallery_image, "Full Image: " + classification_result_str))
        
    return annotated_frame, cropped_images_with_labels, final_prediction_status_html


# --- Streamlit Application Layout ---
st.set_page_config(
    page_title="YOLOv8 + EfficientNetV2S for Waste Classification",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("♻️ YOLOv8 Object Detection & EfficientNetV2S Classification")
st.markdown(
    """
    This application performs real-time object detection using YOLOv8, 
    crops the detected objects, and then classifies each cropped object 
    using an EfficientNetV2S model. If no objects are detected by YOLO, 
    the entire frame will be classified by the EfficientNetV2S model. 
    """
)
st.markdown(f"**YOLOv8 is configured to detect:** {', '.join(CLASS_NAMES_YOLO)}")
st.markdown(f"**The classification model predicts categories:** {', '.join(CLASS_NAMES_CLASSIFICATION)}")

st.sidebar.header("Input Options")
input_option = st.sidebar.radio("Select Input Source:", ("Upload Image", "Webcam"))

annotated_image_placeholder = st.empty()
gallery_placeholder = st.empty()
prediction_status_placeholder = st.empty()

if input_option == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_np = cv2.imdecode(file_bytes, 1) # Decode to BGR
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB) # Convert to RGB

        with st.spinner("Processing image..."):
            annotated_frame, cropped_images_with_labels, final_prediction_status_html = \
                detect_and_classify_objects_streamlit(image_np)
            
            annotated_image_placeholder.image(annotated_frame, caption="Detected Objects (Full Frame)", use_column_width=True)
            
            # Prepare images for Streamlit gallery (just the image arrays)
            gallery_images = [img for img, _ in cropped_images_with_labels]
            gallery_captions = [label for _, label in cropped_images_with_labels]

            if gallery_images:
                gallery_placeholder.header("Cropped Image Classification")
                gallery_placeholder.image(gallery_images, caption=gallery_captions, use_column_width=True)
            else:
                gallery_placeholder.write("No cropped images to display.")
            
            prediction_status_placeholder.markdown(final_prediction_status_html, unsafe_allow_html=True)
    else:
        annotated_image_placeholder.info("Please upload an image to start detection and classification.")

elif input_option == "Webcam":
    st.sidebar.warning("Webcam functionality in Streamlit typically requires a secure context (HTTPS) or running locally. Performance might vary.")
    
    # Streamlit's native webcam input is basic and captures a single frame.
    # For a truly 'live' stream, you'd typically need a custom component or a more complex setup.
    # This example demonstrates processing individual frames from a "Capture" button.
    st.sidebar.markdown("---")
    st.sidebar.markdown("Click 'Capture Frame' to process a single frame from your webcam.")
    
    run_webcam = st.sidebar.checkbox("Start Webcam Capture", False)
    
    FRAME_WINDOW = annotated_image_placeholder.image([])
    GALLERY_DISPLAY = gallery_placeholder.empty()
    STATUS_DISPLAY = prediction_status_placeholder.empty()

    if run_webcam:
        cap = cv2.VideoCapture(0) # 0 for default webcam

        if not cap.isOpened():
            st.error("Error: Could not open webcam.")
            run_webcam = False # Stop trying to run if webcam fails
        else:
            st.sidebar.info("Webcam started. Capturing frames...")
            while run_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to grab frame from webcam. Retrying...")
                    time.sleep(0.1)
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                annotated_frame, cropped_images_with_labels, final_prediction_status_html = \
                    detect_and_classify_objects_streamlit(frame_rgb)
                
                FRAME_WINDOW.image(annotated_frame, caption="Detected Objects (Full Frame)", use_column_width=True)

                gallery_images = [img for img, _ in cropped_images_with_labels]
                gallery_captions = [label for _, label in cropped_images_with_labels]

                if gallery_images:
                    GALLERY_DISPLAY.header("Cropped Image Classification")
                    GALLERY_DISPLAY.image(gallery_images, caption=gallery_captions, use_column_width=True)
                else:
                    GALLERY_DISPLAY.write("No cropped images to display.")
                
                STATUS_DISPLAY.markdown(final_prediction_status_html, unsafe_allow_html=True)
                
                # Update checkbox state if user unchecks it
                run_webcam = st.sidebar.checkbox("Start Webcam Capture", run_webcam)
                
                time.sleep(0.1) # Small delay to prevent overwhelming the CPU

            cap.release() # Release webcam when not running
            st.sidebar.info("Webcam capture stopped.")
    else:
        st.info("Webcam capture is currently stopped. Check 'Start Webcam Capture' in the sidebar to begin.")


st.markdown("---")
st.markdown("Developed by Your Name/Team Name")
