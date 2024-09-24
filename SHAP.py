import numpy as np
from PIL import Image
import sys
import os
import onnxruntime as ort
import logging

sys.path.append(os.path.abspath('D:/Repos/yolov8_XAI/Yolov8-XAI-experiment/mygit submodule add https://github.com/username/sub_repo.gitshap/shap'))

import shap


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ort_session = None


def yolo_predict(image):
    global ort_session
    try:
        logging.info(f"Images shape for ONNX model: {image.shape}")
        image = np.transpose(image, (0, 3, 1, 2))  # Convert to channels-first
        # Run the ONNX model
        ort_inputs = {ort_session.get_inputs()[0].name: image}
        ort_outs = ort_session.run(None, ort_inputs)

        # Extract the segmentation masks or other relevant outputs
        masks = ort_outs[0]  # Adjust based on your model's output structure
        return masks
    except Exception as e:
        logging.error(f"Error during ONNX model prediction: {e}")
        raise


def create_explainer():
    global ort_session
    try:
        # Create a masker for the images
        masker = shap.maskers.Image("inpaint_telea", (1024, 1024, 3))
        logging.info("Masker created successfully.")

        # Create an explainer using the custom prediction function
        explainer = shap.Explainer(lambda x: yolo_predict(x), masker)
        logging.info("Explainer created successfully.")
        return explainer
    except Exception as e:
        logging.error(f"Error creating masker or explainer: {e}")
        raise


def load_onnx_model(onnx_model_path):
    try:
        session = ort.InferenceSession(onnx_model_path)
        logging.info(f"ONNX model loaded successfully from {onnx_model_path}")
        return session
    except Exception as e:
        logging.error(f"Error loading ONNX model: {e}")
        raise


def explain_image_with_shap(image_path, onnx_model_path):
    global ort_session
    try:
        # Load the ONNX model
        ort_session = load_onnx_model(onnx_model_path)

        # Create an explainer using the custom prediction function
        explainer = create_explainer()

        # Load and preprocess the image
        img = Image.open(image_path)
        img = img.resize((1024, 1024))
        image = np.array(img, dtype=np.float32)
        image_with_batch = np.expand_dims(image, axis=0)  # Add batch dimension
        logging.info(f"Image with batch shape: {image_with_batch.shape}")

        # Generate SHAP values
        try:
            shap_values = explainer(image_with_batch, max_evals=2, outputs=shap.Explanation.argsort.flip[:1])
            # shap_values = explainer(image, max_evals=2, outputs=shap.Explanation.argsort.flip[:1])
            logging.info("SHAP values generated successfully.")
        except Exception as e:
            logging.error(f"Error generating SHAP values: {e}")
            raise

        # Save the explanation
        shap.save_html("shap_explanation.html", shap_values)
        logging.info("SHAP explanation saved successfully.")

    except Exception as e:
        logging.error(f"Error in explain_image_with_shap: {e}")
        raise


# Example usage
explain_image_with_shap(
    "D:/Repos/yolov8_XAI/Yolov8-XAI-experiment/data/gtFine/images/test/bielefeld"
    "/bielefeld_000000_000321_leftImg8bit.png",
    "D:/Repos/yolov8_XAI/Yolov8-XAI-experiment/models/best-14870.onnx")
