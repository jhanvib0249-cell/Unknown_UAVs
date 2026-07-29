# ====================================================================
# RUN THIS IN A SEPARATE CELL FIRST TO INSTALL DEPENDENCIES:
!pip install ultralytics kagglehub pyyaml gradio opencv-python
# ====================================================================

import os
import glob
import yaml
import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO

# kagglehub is used to download the public dataset without needing an API key
import kagglehub

# --- GLOBAL MODEL CACHE ---
# Caching the model prevents the system from reloading it on every single video frame,
# which would cause severe lag or memory crashes during live video streaming.
global_model = None

def get_model():
    """Loads the custom trained model if it exists, otherwise falls back to the default YOLOv8n."""
    global global_model
    model_path = "runs/detect/uav_detector/weights/best.pt"

    # Only load the model if it hasn't been loaded into memory yet
    if global_model is None:
        if os.path.exists(model_path):
            print("Loading custom trained UAV model...")
            global_model = YOLO(model_path)
        else:
            print("Custom model not found. Falling back to default YOLOv8n...")
            global_model = YOLO("yolov8n.pt")

    return global_model

# --- CORE FUNCTIONS ---

def train_drone_detector_fast():
    """Downloads the dataset, configures the YAML, and trains YOLOv8."""
    try:
        print("=== STEP 1: Downloading Dataset ===")
        # Download dataset from Kaggle directly
        dataset_path = kagglehub.dataset_download("muki2003/yolo-drone-detection-dataset")

        # Find the data.yaml file in the downloaded folder
        yaml_files = glob.glob(os.path.join(dataset_path, "**", "*.yaml"), recursive=True)
        if not yaml_files:
            return "Error: data.yaml file not found in the dataset folder."

        orig_yaml_path = yaml_files[0]
        dataset_dir = os.path.dirname(orig_yaml_path)

        # Read the original YAML configuration
        with open(orig_yaml_path, "r") as f:
            yaml_content = yaml.safe_load(f)

        # Update the dataset paths to match the Colab/local environment
        yaml_content["path"] = dataset_dir.replace("\\", "/")
        yaml_content["train"] = "train"
        yaml_content["val"] = "valid"

        # Save the updated configuration to a writable local file
        custom_yaml_path = os.path.join(os.getcwd(), "custom_data.yaml")
        with open(custom_yaml_path, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

        print("=== STEP 2: Initializing Model ===")
        last_checkpoint = "runs/detect/uav_detector/weights/last.pt"

        # Resume training if interrupted, otherwise start fresh
        if os.path.exists(last_checkpoint):
            model = YOLO(last_checkpoint)
            resume_flag = True
        else:
            model = YOLO("yolov8n.pt")
            resume_flag = False

        print("=== STEP 3: Starting Training ===")
        model.train(
            data=custom_yaml_path,
            epochs=80,         # Set to 3 for a fast test. Increase to 50+ for real accuracy.
            imgsz=640,        # Standard YOLO resolution
            patience=15,
            batch=16,
            name="uav_detector",
            exist_ok=True,
            resume=resume_flag
        )

        # Reset the global cache so the UI uses the newly trained weights immediately
        global global_model
        global_model = None

        return "Training complete! Weights saved at: runs/detect/uav_detector/weights/best.pt"

    except Exception as e:
        return f"Training failed: {str(e)}"

def detect_uav_image(image):
    """Runs inference on a single uploaded static image."""
    if image is None:
        return None, "Please upload an image first."

    try:
        model = get_model()
        # Run prediction
        results = model.predict(source=image, conf=0.25)

        # Plot the bounding boxes on the image
        for result in results:
            annotated_frame = result.plot()
            # YOLO returns BGR color format, but Gradio requires RGB
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            return annotated_frame_rgb, "Detection successful!"

        return image, "No predictions returned."
    except Exception as e:
        return image, f"Error during detection: {str(e)}"

def detect_uav_video_frame(frame):
    """Runs inference on a continuous real-time video stream."""
    if frame is None:
        # Create a visual placeholder if the camera is blocked, denied, or still loading
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Camera Offline / Blocked", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 50, 50), 2)
        cv2.putText(placeholder, "Please allow camera access in browser", (90, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        return placeholder

    try:
        model = get_model()
        # verbose=False prevents the console from being spammed 30 times a second
        results = model.predict(source=frame, conf=0.25, verbose=False)

        # Plot the bounding boxes on the frame
        annotated_frame = results[0].plot()

        # Convert BGR to RGB for the browser
        return cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    except Exception:
        # If an error occurs (e.g., frame corruption), safely return the original frame
        return frame


# --- GRADIO UI LAYOUT ---

# Using a clean theme
with gr.Blocks(theme=gr.themes.Soft()) as kshiti_ui:
    gr.Markdown(
        """
        # 🚁 Real-Time UAV & Drone Detector
        Train a custom YOLOv8 model, test it on images, and run it on a live webcam feed.
        """
    )

    with gr.Tab("1. Live Video (Webcam)"):
        gr.Markdown(
            "### 📷 Real-Time Detection Feed\n"
            "*Troubleshooting: If your camera does not load, ensure you opened the public `share=True` link ending in `.gradio.live`.*"
        )

        with gr.Row():
            # streaming=True creates a direct pipeline to the detect_uav_video_frame function
            live_input = gr.Image(sources=["webcam"], streaming=True, label="Live Camera Input")
            live_output = gr.Image(label="AI Detection Stream")

        # Connect the continuous input to the output function
        live_input.stream(fn=detect_uav_video_frame, inputs=live_input, outputs=live_output)

    with gr.Tab("2. Detect Image"):
        gr.Markdown("Upload an image to detect UAVs using the trained model.")

        with gr.Row():
            img_input = gr.Image(label="Upload Test Image")
            img_output = gr.Image(label="Detection Result")

        status_text = gr.Textbox(label="Status", interactive=False)
        detect_btn = gr.Button("Run Detection", variant="primary")

        detect_btn.click(fn=detect_uav_image, inputs=img_input, outputs=[img_output, status_text])

    with gr.Tab("3. Train Model"):
        gr.Markdown("Download the dataset and initiate a fast 3-epoch YOLOv8 training session.")

        train_output = gr.Textbox(label="Training Status Log", lines=3, interactive=False)
        train_btn = gr.Button("Start Training", variant="primary")

        train_btn.click(fn=train_drone_detector_fast, inputs=None, outputs=train_output)

# Launch the UI
# share=True creates the public link required to view it outside the Colab cell
if __name__ == "__main__":
    kshiti_ui.launch(debug=True, share=True)