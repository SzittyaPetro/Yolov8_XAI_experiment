from ultralytics import YOLO

# Load the pretrained model
model = YOLO("./models/best-box.pt")

# Run evaluation on the test set
metrics = model.val(
    data="./descriptors/gtFine_descriptor.yaml",
    split='val',  # Specify the test split
    project="yolov8",
    name="test_evaluation",
    batch=4,
    imgsz=1024,
    save_json=True,
    device='0',
    half=False,
    conf=0.25,
    iou=0.6,
)

# Print the evaluation metrics
print("Evaluation Metrics:")
print(f"Precision: {metrics['precision']}")
print(f"Recall: {metrics['recall']}")
print(f"mAP: {metrics['mAP']}")