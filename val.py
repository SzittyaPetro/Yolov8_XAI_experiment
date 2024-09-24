import comet_ml
import ultralytics
from ultralytics import YOLO
# Load a model
import torch

model = YOLO("yolov8m-seg.pt")  # load a pretrained model (recommended for training)
comet_ml.init()
# train the model
if __name__ == '__main__':
    metrics = model.val(
        data="gtFine_descriptor.yaml",
        project="yolov8",
        name="test",
        batch=8,
        imgsz=1024,
        save_json=True,
        device='0',
        half=False,
        conf=0.25,
        iou=0.6,
    )
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
