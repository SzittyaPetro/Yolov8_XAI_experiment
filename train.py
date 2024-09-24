import comet_ml
from ultralytics import YOLO
# Load a model
import torch
model = YOLO("yolov8m-seg.pt")  # load a pretrained model (recommended for training)
comet_ml.init()
# train the model
if __name__ == '__main__':
    results = model.train(
        data = "gtFine_descriptor.yaml",
        project = "yolov8",
        name = "test",
        batch = 4,
        epochs = 10,
        imgsz = 1024,
        device='0'
)
