import comet_ml
import ultralytics
from ultralytics import YOLO
# Load a model
import torch

model = YOLO("./models/best-box_ego_positive.pt")  # load a pretrained model (recommended for training)
comet_ml.login()
# train the model
if __name__ == '__main__':
    metrics = model.val(
        data="./descriptors/gtFine_descriptor.yaml",
        project="yolov8",
        name="test",
        batch=4,
        imgsz=1024,
        save_json=True,
        device='0',
        half=False,
        conf=0.25,
        iou=0.6,
    ) # map75
