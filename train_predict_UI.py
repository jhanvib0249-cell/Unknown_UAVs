# ====================================================================
# RUN THIS IN A SEPARATE CELL FIRST:
# !pip install ultralytics kagglehub pyyaml gradio opencv-python
# ====================================================================

import os
import glob
import yaml
import kagglehub
import cv2
import gradio as gr
from ultralytics import YOLO

def train_drone_detector_fast():
    """Handles the dataset setup and model training."""
    try:
        print("=== STEP 1: Setting up Dataset ===")
        # Download dataset from Kaggle
        dataset_path = kagglehub.dataset_download("muki2003/yolo-drone-detection-dataset")
        
        # Find the original data.yaml file
        yaml_files = glob.glob(os.path.join(dataset_path, "**", "*.yaml"), recursive=True)
        if not yaml_files:
            return "Error: data.yaml file not found in dataset folder."

        orig_yaml_path = yaml_files[0]
        dataset_dir = os.path.dirname(orig_yaml_path)

        # Read from the read-only dataset YAML
        with open(orig_yaml_path, "r") as f:
            yaml_content = yaml.safe_load(f)

        # Update paths inside the dictionary
        yaml_content["path"] = dataset_dir.replace("\\", "/")
        yaml_content["train"] = "train"
        yaml_content["val"] = "valid"

        # Save as a NEW file in the current writable working directory
        custom_yaml_path = os.path.join(os.getcwd(), "custom_data.yaml")
        with open(custom_yaml_path, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        # Initializing Model
        last_checkpoint = "runs/detect/uav_detector/weights/last.pt"
        if os.path.exists(last_checkpoint):
            model = YOLO(last_checkpoint)
            resume_flag = True
        else:
            model = YOLO("yolov8n.pt")
            resume_flag = False

        # Starting Training
        model.train(
            data=custom_yaml_path,
            epochs=3,
            imgsz=640,
            batch=16,
            name="uav_detector",
            exist_ok=True,
            resume=resume_flag
        )
        return "Training complete! Weights are saved at runs/detect/uav_detector/weights/best.pt"
    
    except Exception as e:
        return f"Training failed: {str(e)}"

def detect_uav(image):
    """Handles inference on user-uploaded images."""
    if image is None:
        return None, "Error: Please upload an image first."

    model_path = "runs/detect/uav_detector/weights/best.pt"
    
    if not os.path.exists(model_path):
        return image, "Error: Trained model not found. Please run the 'Train Model' tab first."
    
    try:
        # Load trained model
        model = YOLO(model_path)
        
        # Run prediction on uploaded file(s) with confidence 0.25
        results = model.predict(source=image, conf=0.25)
        
        for result in results:
            annotated_frame = result.plot()
            # Convert BGR to RGB for Gradio UI display compatibility
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            return annotated_frame_rgb, "Detection successful!"
        
        return image, "No predictions returned."
    
    except Exception as e:
        return image, f"Detection error: {str(e)}"

# === Gradio UI Layout ===
with gr.Blocks(theme=gr.themes.Soft()) as kshiti_ui:
    gr.Markdown("# 🚁 KSHITI UAV - YOLOv8 Detector Interface")
    gr.Markdown("Train your custom drone detection model and test it on new images instantly.")
    
    with gr.Tab("1. Detect UAV"):
        gr.Markdown("Upload an image to detect UAVs using the trained model.")
        with gr.Row():
            img_input = gr.Image(label="Upload Test Image")
            img_output = gr.Image(label="Detection Result")
        
        status_text = gr.Textbox(label="Status")
        detect_btn = gr.Button("Run Detection", variant="primary")
        
        detect_btn.click(fn=detect_uav, inputs=img_input, outputs=[img_output, status_text])
        
    with gr.Tab("2. Train Model"):
        gr.Markdown("Download the dataset and initiate a fast 3-epoch YOLO training session.")
        train_output = gr.Textbox(label="Training Status Log", lines=3)
        train_btn = gr.Button("Start Training", variant="primary")
        
        train_btn.click(fn=train_drone_detector_fast, inputs=None, outputs=train_output)

# Launch the UI in the Colab output cell
if __name__ == "__main__":
    kshiti_ui.launch(debug=True, share=True)