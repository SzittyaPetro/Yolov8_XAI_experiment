import numpy as np
from PIL import Image
import sys
import os
import onnxruntime as ort
import logging

# Add the path to the sys path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import myshap.shap.shap as shap


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ort_session = None


def yolo_predict(image):
    global ort_session
    try:
        logging.info(f"Images shape for ONNX model: {image.shape}")
        # Run the ONNX model
        ort_inputs = {ort_session.get_inputs()[0].name: image}
        ort_outs = ort_session.run(None, ort_inputs)

        # Extract the relevant outputs for YOLOv8m
        predictions = ort_outs[0]  # Adjust based on your model's output structure
        return predictions
    except Exception as e:
        logging.error(f"Error during ONNX model prediction: {e}")
        raise


def create_explainer():
    global ort_session
    try:
        # Create a masker for the images
        masker = shap.maskers.Image("inpaint_telea", (640, 640, 3))
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
        img = img.resize((640, 640))  # Resize to YOLOv8m input size
        image = np.array(img, dtype=np.float32)
        image /= 255.0  # Normalize
        image = image.transpose(2, 0, 1)  # Convert to channels-first
        image_with_batch = np.expand_dims(image, axis=0)  # Add batch dimension
        logging.info(f"Image with batch shape: {image_with_batch.shape}")

        # Generate SHAP values
        try:
            #shap_values = explainer.shap_values(np.ones((1, 50)), nsamples=1000)
            shap_values = explainer(image_with_batch, max_evals=2)
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



