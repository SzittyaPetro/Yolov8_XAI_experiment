from ultralytics import YOLO
import os

# Load a model
model = YOLO('yolov8m-seg.pt')  # load an official model

# Predict with the model
results = model.predict('D:/Uni/Yolov8/data/gtFine/val/Img/munster/', save=True, imgsz=2040,
                        conf=0.5)  # predict on an image

# Save the predicted image
output_path = 'D:/Uni/Yolov8/runs/segmentpredicted_image.png'
for i,result in enumerate(results):
    result.save(save_dir=os.path.dirname(output_path), save_name=f"{os.path.splitext(os.path.basename(output_path))[0]}_{i}.png")

# Check if the file was saved successfully
if os.path.exists(output_path):
    print(f"Predicted image saved successfully at {output_path}")
else:
    print("Error: Failed to save predicted image.")
