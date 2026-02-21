import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO

# Load YOLO model once
model = YOLO("yolov8x.pt")

def detect_objects(image):
    # Convert to OpenCV format
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Run inference
    results = model(image)

    # Get annotated image
    annotated_image = results[0].plot()

    # Convert back to RGB for display
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    return annotated_image

# Gradio UI
demo = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="numpy", label="Upload an Image"),
    outputs=gr.Image(type="numpy", label="Detected Objects"),
    title="YOLO Object Detection",
    description="Upload an image to detect objects using YOLOv8."
)

if __name__ == "__main__":
    demo.launch()
