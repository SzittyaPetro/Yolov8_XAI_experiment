# Description: This script is used to generate explanations for the YOLOv8 model using EigenCAM and LIME.
from pathlib import Path
import cv2
import numpy as np
import logging
from torch.backends.mkl import verbose
from ultralytics import YOLO
from MEigenCAM.yolov8_cam import EigenCAM, EigenGradCAM
from MEigenCAM.yolov8_cam.utils.image import show_cam_on_image  # noqa
from LIME import lime_result, Yolo_sum, Yolo_multi
from comet_ml import API

from EigenGradCAM import yolov8_heatmap
from utils.general import save_images

from SHAP import explain_image_with_shap

# Initialize logger
logging.basicConfig(level=logging.INFO)


def download_model(api_key, workspace, project, experiment_key, model_file_name) -> str:
    """
    Download the model file from Comet.ml.

    Parameters
    ----------
    api_key : str
        Comet API key.
    workspace : str
        Comet workspace.
    project : str
        Comet project.
    experiment_key : str
        Comet experiment key.
    model_file_name : str
        Name of the model file.

    Returns
    -------
    str
        Path to the downloaded model file.
    """
    # Initialize Comet API
    comet_api = API(api_key=api_key)

    # Specify the experiment
    experiment = comet_api.get(workspace, project, experiment_key)

    # Get the asset id of the model file
    assets = experiment.get_asset_list()
    model_asset_id = None
    for asset in assets:
        if asset['fileName'] == model_file_name:
            model_asset_id = asset['assetId']
            break

    if model_asset_id is None:
        raise ValueError("Model file not found in the experiment assets.")

    # Get the asset data
    asset_data = experiment.get_asset(model_asset_id)

    model_file_path = f"models/{model_file_name}"

    # Write the asset data to a file
    with open(model_file_path, 'wb') as f:
        f.write(asset_data)

    return model_file_path


def process_eigen_cam(file_path,output_dir, model)-> None:
    """
    Process the image and generate CAMs for each class in the image using EigenCAM.


    Parameters
    ----------
    file_path : Path
        Path to the image file.
    output_dir : Path
        Output directory.
    model : YOLO
        YOLO model.

    Returns
    -------
    None
    """
    # Load the image
    img = cv2.imread(str(file_path))

    # Resize the image
    img = cv2.resize(img, (640, 640))

    # Copy the image
    rgb_img = img.copy()

    # Normalize the image
    img = np.float32(img) / 255
    num_classes = model.model.model[-1].nc  # Get the number of classes from the model

    for i in range(2, 7):
        target_layers = [model.model.model[-i]]
        for class_id in range(num_classes):

            # Create EigenCAM object
            cam = EigenCAM(model, target_layers, task='od')


            # Check if class_id is within the range of cam(rgb_img)'s first dimension
            if class_id < cam(rgb_img).shape[0]:
                grayscale_cam = cam(rgb_img)[class_id, :, :]  # Get the CAM for the specific class
                cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

                # Create output directory for each class
                output = output_dir / f"{file_path.name}layer-{i}"
                output.mkdir(parents=True, exist_ok=True)

                # Save the CAM image
                output_file_path = output / f"{file_path.name}_object({class_id})_heatmap.jpg"
                success = cv2.imwrite(str(output_file_path), cam_image)
                if not success:
                    logging.error(f"Failed to save image at {output_file_path}")
            else:
                break

def process_EigenGradCAM(file_path, output_dir, model)->None:
    """
    Process the image and generate EigenGradCAM explanations.
    Parameters
    ----------
    file_path
    output_dir
    model

    Returns
    -------

    """

    from MEigenCAM.yolov8_cam.utils.model_targets import ClassifierOutputTarget
    from MEigenCAM.yolov8_cam.eigen_grad_cam import EigenGradCAM
    img = cv2.imread(file_path)

    # Resize the image
    img = cv2.resize(img, (640, 640))

    # Copy the image
    rgb_img = img.copy()

    target_layers = [model.layer4[-1]]
    input_tensor = rgb_img# Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!

    # We have to specify the target we want to generate the CAM for.
    targets = [ClassifierOutputTarget(281)]

    # Construct the CAM object once, and then re-use it on many images.
    with EigenGradCAM(model=model, target_layers=target_layers) as cam:
      # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
      grayscale_cam = cam(input_tensor=input_tensor, eigen_smooth=True)
      # In this example grayscale_cam has only one image in the batch:
      grayscale_cam = grayscale_cam[0, :]
      visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

      # Save the visualized image
      output_dir = Path(output_dir) / "EigenGradCAM"
      output_dir.mkdir(parents=True, exist_ok=True)
      output_file_path = output_dir / f"{file_path.stem}_eigen_grad_cam.jpg"
      cv2.imwrite(str(output_file_path), visualization)


def process_image(file_path: Path,output_dir: Path, model,weight_path,lime=False,shap=False,eigengradcam=True, eigencam=False)-> None:
    """
    Process the image and generate explanations using both EigenCAM and LIME. The explanations are saved in the explain_output directory.

    Parameters
    ----------
    eigencam : bool
        Flag for using EigenCAM explanations.
    eigengradcam : bool
        Flag for using EigenGradCAM explanations.
    shap : bool
        Flag for using SHAP explanations.
    lime : bool
        Flag for using LIME explanations.
    Eig
    file_path : Path
        Path to the image file.
    output_dir : Path
        Output directory.
    model : YOLO
        YOLO model.

    Returns
    -------
    None
    """
    if lime:
        lime_output= output_dir/ "LIME/"
        lime_output.mkdir(parents=True, exist_ok=True)
        # Process LIME
        lime_result(model=model,model_name=model.model_name, batch_predict=Yolo_sum, output_paths=lime_output, paths=[file_path], num_samples=500,num_features=10)
        lime_result(model=model,model_name=model.model_name, batch_predict=Yolo_multi, output_paths=lime_output, paths=[file_path],
                num_samples=500, num_features=20)
    if shap:
        shap_output= output_dir/ "SHAP/"
        shap_output.mkdir(parents=True, exist_ok=True)
        explain_image_with_shap(file_path,"D:/Repos/yolov8_XAI/Yolov8_XAI_experiment/models/best-box.onnx")
        if not Path("D:/Repos/yolov8_XAI/Yolov8_XAI_experiment/models/best-box.onnx").exists():
            model.export( format="onnx" )
        explain_image_with_shap(file_path, "D:/Repos/yolov8_XAI/Yolov8_XAI_experiment/models/best-box.onnx")
    if eigengradcam:
        gradcam_output= output_dir/ f"EigenGradCAM/{file_path.parts[-2]}"
        gradcam_output.mkdir(parents=True, exist_ok=True)
        eigengradcammodel = yolov8_heatmap(
            #weight="D:/Repos/yolov8_XAI/Yolov8_XAI_experiment/models/best-14870.pt",
            weight=weight_path,
            method="EigenGradCAM",
        )
        imagelist = eigengradcammodel(
            img_path=str(file_path),

        )
        save_images(imagelist,output_dir=gradcam_output)
    if eigencam:
        eigencam_output= output_dir/ "EigenCAM/"
        eigencam_output.mkdir(parents=True, exist_ok=True)
        # Process EigenCAM
        process_eigen_cam(file_path,eigencam_output, model)



def main(arguments):
    """
    Main function of YOLOv8 XAI.

    Parameters
    ----------
    arguments : argparse.Namespace
        Parsed command line arguments.

    Returns
    -------
    None
    """
    # Specify whether to use the local model or download from Comet.ml
    if arguments.use_remote_model:
        model_file_path = download_model("gpludwOtJhDm2xDtgBjk5o9LT", "szittyapetro", "yolov8", "defensive-wok",
                                         "best.pt")
        model = YOLO(model_file_path)
    else:
        model = YOLO("models/best-box.pt", task="detect", verbose=False)
    # Specify the output directory
    output_dir = Path("./output/explain")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Specify the directory
    dir_path = Path('data/gtFine-seg/images/test/bonn')

    # Get a list of all files in the directory
    file_names = dir_path.iterdir()

    # Process each picture in the directory
    for file_name in file_names:
        if not file_name.is_file() or file_name.suffix not in [".png", ".jpg"]:
            continue
        # Construct the full file path
        file_path = file_name

        if not file_path.exists():
            logging.warning(f"File {file_path} does not exist. Skipping.")
            continue

        logging.info(f"Processing file {file_path}...")
        process_image(file_path,output_dir, model,"models/best-box.pt")


def parse_args():
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """

    import argparse
    parser = argparse.ArgumentParser(description="YOLOv8 XAI")
    parser.add_argument("--use-remote-model", action="store_true",
                        help="Use the local model instead of downloading from Comet.ml")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    # Run the main function
    main(args)
