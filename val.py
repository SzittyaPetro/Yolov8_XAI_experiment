import comet_ml
import ultralytics
from ultralytics import YOLO
# Load a model
import torch

model = YOLO("./models/best-box.pt")
comet_ml.login()
# train the model
if __name__ == '__main__':
    metrics = model.val(
        data="./descriptors/gtFine_descriptor.yaml",
        project="yolov8",
        name="test",
        batch=16,
        imgsz=1024,
        save_json=True,
        device='0',
        half=False,
        conf=0.25,
        iou=0.8,
    ) # map75
    print("Evaluation Metrics:")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"mAP: {metrics['mAP']}")
    print(f"IoU: {metrics['IoU']}")

