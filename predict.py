from ultralytics import YOLO
import os

# Load a model
model = YOLO('./models/best-box.pt')  # load an official model

# Predict with the model
results = model.predict('D:/Repos/yolov8_XAI/Yolov8_XAI_experiment/data/gtFine/images/test/bonn/bonn_000035_000019_leftImg8bit.png', save=True, imgsz=2040,
                        conf=0.5)  # predict on an image

# Save the predicted image
output_path = 'D:/Uni/Yolov8/output/predict'
for i, result in enumerate(results):
    result.save(filename=f"{os.path.splitext(os.path.basename(output_path))[0]}_{i}.png")
# Check if the file was saved successfully
if os.path.exists(output_path):
    print(f"Predicted image saved successfully at {output_path}")
else:
    print("Error: Failed to save predicted image.")
