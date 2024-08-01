import streamlit as st
import torch
from PIL import Image
import cv2
import numpy as np

# Load your YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='bestonlyeye',device='cpu')  # Replace with your model path

def detect_objects(image, confidence):
    results = model(image)
    results.xyxy[0] = results.xyxy[0][results.xyxy[0][:, 4] > confidence]
    return results.xyxy[0]

# Streamlit app
st.title("Object Detection App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
confidence = st.slider('Confidence', min_value=0.0, max_value=1.0, value=0.3)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    
    # Perform object detection
    detections = detect_objects(image_np, confidence)

    # Display results
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if len(detections) > 0:
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image_np, f"{int(cls)} - {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        st.image(image_np, caption='Detections', use_column_width=True, channels='BGR')
    else:
        st.write("No detections found.")
