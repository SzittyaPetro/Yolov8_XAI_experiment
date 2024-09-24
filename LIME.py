import cv2
from lime import lime_image
from matplotlib import pyplot as plt
from ultralytics import YOLO

import numpy as np
from PIL import Image
from skimage.segmentation import mark_boundaries


def load_images_from_paths(image_paths):
    """
    Load images from file paths
    :param image_paths:
    :return:
    """
    # Define a function to load images from file paths
    images = []
    for path in image_paths:
        image = Image.open(path)
        images.append(np.array(image))
    return images


def add_border_to_image(im, border_type=cv2.BORDER_CONSTANT, value=[0, 0, 0]):
    """
    Add border to the image with specified border sizes and type.
    :param im: Source image
    :param border_type: Type of the border (default is cv2.BORDER_CONSTANT)
    :param value: Border color (default is black)
    :return: Image with border added
    """
    #if im.ndim > 2:
    #    raise ValueError("Source image must have 2 or fewer dimensions")

    row, col = im.shape[:2]
    bottom = im[row-2:row, 0:col]
    mean = cv2.mean(bottom)[0]

    border_size = 10
    return cv2.copyMakeBorder(
        im,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=[mean, mean, mean]
    )


class YoloLimeExplainer:
    """
    YoloLimeExplainer is a class that provides functionality to explain YOLO model predictions using LIME (Local Interpretable Model-agnostic Explanations).
    """

    def __init__(self, model):
        """
        Initialize LIME explainer and passing of the YOLO model
        """
        self.model = model
        self.explainer = lime_image.LimeImageExplainer()

    def yolo_segmentation_predict(self, images):
        """Used for generating predictions with the given YOLO model"""
        results = self.model(images)
        segmented_images = []
        for result in results:
            segmented_images.append(result.masks.data.cpu().numpy())
        return segmented_images

    def explain_instance(self, image, predict_fn, top_labels=5, hide_color=0, num_samples=1000):
        """
        Explain the instance using LIME for the provided image.
        :param image: The image to be explained.
        :param predict_fn: The prediction function to be used by LIME.
        :param top_labels: The number of top labels to be explained.
        :param hide_color: The color to hide the superpixels.
        :param num_samples: The number of samples to be used by LIME.
        """
        output_dir_path = image.parent / "explain_output" / "LIME"
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_file_path = output_dir_path / f"{image.name}_lime_explanation.png"

        val_images = load_images_from_paths([image])
        image_with_border = add_border_to_image(np.array(val_images[0]))  # Add border

        print(f"Image shape before adding border: {np.array(val_images[0]).shape}")
        print(f"Image shape after adding border: {image_with_border.shape}")

        explanation = self.explainer.explain_instance(image_with_border, predict_fn,
                                                      top_labels=top_labels,
                                                      hide_color=hide_color, num_samples=num_samples)

        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
                                                    hide_rest=True)
        sample_image_with_mask = mark_boundaries(temp / 2 + 0.5, mask)

        plt.imsave(output_file_path, sample_image_with_mask)

    def __call__(self, image, predict_fn, top_labels=5, hide_color=0, num_samples=1000):
        self.explain_instance(image, predict_fn, top_labels, hide_color, num_samples)
