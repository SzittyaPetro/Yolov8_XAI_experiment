import comet_ml
from ultralytics import YOLO

import torch

# train the model

if __name__ == '__main__':
    model = YOLO(model="D:/Repos/yolov8_XAI/Yolov8_XAI_experiment/models/yolov8m.pt")  # load a pretrained model (recommended for training)
    comet_ml.login()
    results = model.train(
        data = "./descriptors/gtFine_descriptor.yaml",
        project = "yolov8",
        name = "bbox",
        batch = 3,
        epochs = 30,
        imgsz = 1024,
        device='0'
)
