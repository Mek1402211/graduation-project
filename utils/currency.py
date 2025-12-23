from ultralytics import YOLO
import cv2
import json
import torch

# Load YOLO model once
try:
   currency_model = YOLO("models/project_model1.pt")
   print("YOLO model loaded successfully.")
except Exception as e:
   print(f"Error loading YOLO model: {e}")
device = 0 if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
def detect_currency(image_path):
    results = currency_model.predict(image_path,conf=0.8, device=device, verbose=False)
    if len(results[0].boxes) == 0:
        return "No currency detected"
    currency = results[0].to_json()
    data = json.loads(currency)
    key_value_pairs = {item["name"]: item["confidence"] for item in data}
    return key_value_pairs