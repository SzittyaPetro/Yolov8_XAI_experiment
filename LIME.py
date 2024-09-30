from pathlib import Path
from lime import lime_image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries
import matplotlib as mpl

mpl.rc('figure', max_open_warning = 0)


detector1= None

# https://github.com/akshay-gupta123/Face-Mask-Detection

def lime_result(model_name,model, batch_predict,output_paths, paths, num_samples=200, num_features=10):
    """
    Generate LIME explanations for a set of images using a specified model and prediction function.

    Parameters
    ----------
    model_name : str
        The name of the model used for generating explanations.
    model : YOLO
        The YOLO model used for predictions.
    batch_predict : function
        The function used to predict the output for a batch of images.
    output_paths : Path
        The path to save the output images.
    paths : list
        List of file paths to the images.
    num_samples : int, optional
        The number of samples to generate for LIME. Default is 200.
    num_features : int, optional
        The number of features to include in the explanation. Default is 10.

    Returns
    -------
    None
    """
    global detector1
    # lime explainer
    explainer = lime_image.LimeImageExplainer()
    detector1 = model
    for i, path in enumerate(paths):
        img = Image.open(path)
        img = img.resize((896,896))

        explanation = explainer.explain_instance(np.array(img), batch_predict, top_labels=5, hide_color=0,
                                                 num_samples=num_samples)

        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True,
                                                    num_features=num_features, hide_rest=False)
        img_boundry1 = mark_boundaries(temp / 255.0, mask)

        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False,
                                                    num_features=num_features, hide_rest=False)
        img_boundry2 = mark_boundaries(temp / 255.0, mask)

        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(img_boundry1)
        axarr[1].imshow(img_boundry2)
        output= f'{output_paths}/{Path(model_name).name}_{Path(path).name}.jpg'
        plt.savefig(output)


def Yolo_sum(images):
    """
    Summarize YOLO model predictions for a batch of images with multiple confidence scores.

    This function processes a list of images using a YOLO model and calculates multiple confidence scores for each image. The results are returned as an array of predictions.

    Parameters
    ----------
    images : list
        List of images to be processed.

    Returns
    -------
    np.ndarray
        Array of predictions, where each prediction is a list containing multiple confidence scores and their complements.
    """
    global detector1
    pred = []
    for image in images:
        sum = 0
        logits = detector1(image)
        if isinstance(logits, list):
            logits = logits[0]  # Assuming the first element is the `Results` object
        boxes = logits.boxes
        if boxes is not None:
            conf_scores = boxes.conf  # Access the confidence scores
            result = np.array(conf_scores)
            for i in result:
                sum += i
            sum = sum / len(result) if len(result) else 0
        pred.append([sum, 1 - sum])
    return np.array(pred)


def Yolo_multi(images):
    """
    Summarize YOLO model predictions for a batch of images with multiple confidence scores.

    This function processes a list of images using a YOLO model and calculates multiple confidence scores for each image. The results are returned as an array of predictions.

    Parameters
    ----------
    images : list
        List of images to be processed.

    Returns
    -------
    np.ndarray
        Array of predictions, where each prediction is a list containing multiple confidence scores and their complements.
    """
    pred = []
    for num, image in enumerate(images):
        results = []
        logits = detector1(image)
        if isinstance(logits, list):
            logits = logits[0]  # Assuming the first element is the `Results` object
        boxes = logits.boxes
        if boxes is not None:
            conf_scores = boxes.conf  # Access the confidence scores
            result = np.array(conf_scores)
            for i in result:
                results.append(i)
                results.append(1 - i)
        pred.append(results)
    np_result = np.zeros((len(images), 100))
    for i in range(len(images)):
        size = len(pred[i]) if len(pred[i]) <= 100 else 100
        np_result[i, :size] = pred[i][:size]
    return np_result
