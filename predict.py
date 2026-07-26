import io
import requests
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

def detect_uav_from_url(image_url: str, model_path: str = "runs/detect/uav_detector/weights/best.pt"):
    """
    Downloads an image from a URL and runs UAV detection using trained YOLOv8 model.
    """
    # 1. Load trained model (falls back to pretrained yolov8n.pt if best.pt is missing)
    try:
        model = YOLO(model_path)
        print(f"Loaded model weights from {model_path}")
    except Exception:
        print(f"Could not find {model_path}. Falling back to default pretrained YOLOv8 model...")
        model = YOLO("yolov8n.pt")

    # 2. Fetch image from URL
    print(f"Fetching image from: {image_url}")
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(image_url, headers=headers, stream=True)
    
    if response.status_code != 200:
        raise ValueError(f"Failed to retrieve image. HTTP Status: {response.status_code}")

    # Convert binary image content to OpenCV format
    img_pil = Image.open(io.BytesIO(response.content)).convert("RGB")
    img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # 3. Perform prediction
    results = model.predict(source=img_cv2, conf=0.4)  # Confidence threshold = 40%

    # 4. Extract detections and save visualization
    for result in results:
        # Plot visual bounding boxes on image
        annotated_frame = result.plot()
        
        # Save output image locally
        output_filename = "uav_detection_result.jpg"
        cv2.imwrite(output_filename, annotated_frame)
        print(f"Detection result saved to '{output_filename}'")

        # Print detection summaries in terminal
        boxes = result.boxes
        print(f"\n--- Detection Summary ---")
        print(f"Total objects detected: {len(boxes)}")
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]
            print(f"- Detected '{class_name}' with {conf * 100:.2f}% confidence")

        # 5. Display image in a pop-up window
        cv2.imshow("UAV Detection Result", annotated_frame)
        print("Press any key on the image window to close it.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Test URL containing a drone/UAV image
    #sample_url = "https://images.unsplash.com/photo-1527977966376-1c8408f9f108?q=80&w=1000&auto=format&fit=crop"
    sample_url = input("Enter the URL of the image to detect UAVs: ")
    detect_uav_from_url(sample_url)