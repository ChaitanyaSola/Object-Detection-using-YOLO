from ultralytics import YOLO

# Load pretrained YOLOv8 model
model = YOLO("yolov8s.pt")

# Train on custom dataset
model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)

print("Training completed successfully!")
