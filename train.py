import comet_ml
from ultralytics import YOLO
# Load a model
import torch
model = YOLO(model="./models/yolov8m.pt")  # load a pretrained model (recommended for training)
comet_ml.login()
# train the model
if __name__ == '__main__':
    results = model.train(
        data = "./descriptors/gtFine_descriptor.yaml",
        project = "yolov8",
        name = "bbox",
        batch = 4,
        epochs = 10,
        imgsz = 1024,
        device='0'
)
