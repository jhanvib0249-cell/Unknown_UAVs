import os
import glob
import kagglehub
from ultralytics import YOLO

def train_drone_detector():
    print("Downloading dataset from Kaggle...")
    # Downloads dataset to local cache directory
    dataset_path = kagglehub.dataset_download("muki2003/yolo-drone-detection-dataset")
    print(f"Dataset downloaded to: {dataset_path}")

    # Search for data.yaml inside the downloaded dataset folder
    yaml_files = glob.glob(os.path.join(dataset_path, "**", "*.yaml"), recursive=True)
    
    if not yaml_files:
        raise FileNotFoundError(
            "data.yaml file not found in the dataset folder. "
            "Ensure the dataset includes YOLO format configuration."
        )
    
    data_yaml_path = yaml_files[0]
    print(f"Found data.yaml at: {data_yaml_path}")

    # Initialize YOLOv8 model (yolov8n = nano, yolov8s = small)
    model = YOLO("yolov8n.pt") 

    # Start training
    print("Starting YOLOv8 training...")
    results = model.train(
        data=data_yaml_path,
        epochs=25,          # Adjust epochs as needed for your hackathon timeline
        imgsz=640,          # Standard image size
        batch=16,           # Lower if running into GPU/CPU memory limits
        name="uav_detector",
        exist_ok=True
    )
    
    print("Training complete!")
    print(f"Trained weights saved to: runs/detect/uav_detector/weights/best.pt")

if __name__ == "__main__":
    train_drone_detector()