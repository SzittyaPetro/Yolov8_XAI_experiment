import logging

import numpy as np
import onnx
from PIL import Image
import onnxruntime as ort
from SHAP import load_onnx_model, yolo_predict


def check_onnx_model(onnx_model_path):
    try:
        # Load the ONNX model
        model = onnx.load(onnx_model_path)
        onnx.checker.check_model(model)
        logging.info(f"ONNX model {onnx_model_path} is valid.")

        # Create an ONNX runtime session
        ort_session = ort.InferenceSession(onnx_model_path)

        # Print input and output names and shapes
        logging.info("Model inputs:")
        for input in ort_session.get_inputs():
            logging.info(f"Name: {input.name}, Shape: {input.shape}, Type: {input.type}")

        logging.info("Model outputs:")
        for output in ort_session.get_outputs():
            logging.info(f"Name: {output.name}, Shape: {output.shape}, Type: {output.type}")

        # Create a dummy input with the correct shape
        input_shape = ort_session.get_inputs()[0].shape
        dummy_input = np.random.rand(*input_shape).astype(np.float32)

        # Run a simple inference
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
        ort_outs = ort_session.run(None, ort_inputs)
        logging.info("Inference ran successfully with dummy input.")
    except Exception as e:
        logging.error(f"Error in check_onnx_model: {e}")
        raise


def validate_yolo_predict(image_path):
    try:

        # Load and preprocess the image
        img = Image.open(image_path)
        img = img.resize((1024, 1024))
        image = np.array(img, dtype=np.float32)
        image = np.transpose(image, (2, 0, 1))  # Convert to channels-first
        image_with_batch = np.expand_dims(image, axis=0)  # Add batch dimension

        # Run the prediction
        masks = yolo_predict( image_with_batch)

        # Print the results
        logging.info(f"Prediction masks shape: {masks.shape}")
        logging.info(f"Prediction masks: {masks}")

    except Exception as e:
        logging.error(f"Error in validate_yolo_predict: {e}")
        raise
