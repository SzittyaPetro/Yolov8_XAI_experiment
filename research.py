
import random
from glob import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shap
import torchvision
import cv2
from utils.general import non_max_suppression, box_iou
device = torch.device('cuda:0')
from IPython.display import clear_output

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


model = torch.load("models/best-box.pt", map_location=device, )['model'].float()


def image_processing(path, img_size, show_image_processing=0):
    img_org = cv2.imread(path, cv2.IMREAD_COLOR)
    img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
    old_img_size = img_org.shape[:2]  # old size is in (height, width) format

    ratio = float(img_size) / max(old_img_size)
    new_size_y, new_size_x = tuple([int(x * ratio) for x in old_img_size])
    img = cv2.resize(img_org, (new_size_x, new_size_y))

    delta_w = img_size - new_size_x
    delta_h = img_size - new_size_y
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    # for making a square image as model expects
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    img = img.astype("float64") / 255
    img_gray = rgb2gray(img)

    if show_image_processing:
        plt.figure(figsize=(15, 15))
        plt.imshow(img_org)
        plt.show()

        fig = plt.figure(figsize=(20, 40))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.imshow(img_gray, cmap='gray', vmin=0, vmax=1)
        plt.show()
    return img_org, img, img_gray

#img_org,img_pre,img_gray=image_processing('./data/gtFine/images/test/bonn/bonn_000035_000019_leftImg8bit.png',10*32,1)


def model_processing(img, confidence, iou, show_yolo_result=0):
    torch_image = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).to(device).unsqueeze(0)
    prediction = model(torch_image.float())
    output = non_max_suppression(prediction[0], conf_thres=confidence, iou_thres=iou)
    if show_yolo_result:
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(1, 1, 1)

        plt.imshow(img)
        for i, detection in enumerate(output[0].cpu().numpy()):
            label = f"{model.names[int(detection[5])]} {detection[4]:0.1%} ({i})"
            bbox = patches.Rectangle(detection[:2], detection[2] - detection[0], detection[3] - detection[1],
                                     linewidth=3, edgecolor='g', facecolor='none')
            plt.text(detection[0], detection[1], label, color="red")
            ax.add_patch(bbox)
        plt.show()

    return output, prediction



#output,prediction=model_processing(img_pre,0.3,0.3,1)


class CastNumpy(torch.nn.Module):
    def __init__(self):
        super(CastNumpy, self).__init__()

    def forward(self, image):
        # In the forward function we accept the inputs and cast them to a pytorch tensor
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).to(device)
        if image.ndimension() == 3:
            image = image.unsqueeze(0)
        return image


numpy2torch_converter = CastNumpy()


class OD2Score(torch.nn.Module):
    def __init__(self, target, conf_thresh=0.01, iou_thresh=0.5):
        super(OD2Score, self).__init__()
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.target = torch.tensor(target, device=device)

    def forward(self, x):
        # In the forward function we accept the predictions and return the score for a selected target of the box
        score_best_box = torch.zeros([x[0].shape[0]], device=device)

        for idx, img in enumerate(x[0]):
            img = img.unsqueeze(0)

            output = non_max_suppression(img, conf_thres=self.conf_thresh, iou_thres=self.iou_thresh)
            if output and output[0] is not None:
                correct_class_predictions = output[0][..., 5] == self.target[5]
                correctly_labeled_boxes = output[0][correct_class_predictions]

                if correctly_labeled_boxes.shape[0]:
                    iou_with_target, _idx = box_iou(correctly_labeled_boxes[:, :4],
                                                    self.target.unsqueeze(0)[:, :4]).max(1)
                    index_best_box_in_correct_class = torch.argmax(iou_with_target)
                    index_best_box_in_output = torch.where(output[0][..., 5] == self.target[5])[0][
                        index_best_box_in_correct_class]

                    score_best_box[idx] = output[0][index_best_box_in_output][4] * iou_with_target[
                        index_best_box_in_correct_class]

        return score_best_box.cpu().numpy()


class SuperPixler(torch.nn.Module):
    def __init__(self, image, super_pixel_width):
        super(SuperPixler, self).__init__()

        self.image = image.transpose(2, 0, 1)  # model expects images in BRG, not RGB, so transpose color channels
        self.mean_color = self.image.mean()
        self.image = np.expand_dims(self.image, axis=0)
        self.image_width = image.shape[1]
        self.super_pixel_width = super_pixel_width

    def forward(self, x):
        # In the forward step we accept the super pixel masks and transform them to a batch of images
        pixeled_image = np.repeat(self.image.copy(), x.shape[0], axis=0)

        for i, super_pixel in enumerate(x.T):
            images_to_pixelate = [bool(p) for p in super_pixel]
            x = (i * self.super_pixel_width // self.image_width) * self.super_pixel_width
            y = i * self.super_pixel_width % self.image_width
            pixeled_image[images_to_pixelate, :, x:x + self.super_pixel_width,
            y:y + self.super_pixel_width] = self.mean_color

        return pixeled_image



def shap_result(img, img_gray, target, target_index, super_pixel_width, img_size, scoring):
    # use Kernel SHAP to explain the detection
    assert (img_size / super_pixel_width) % 1 == 0, "image width needs to be multiple of super pixel width"
    n_super_pixel = int((img.shape[1] / super_pixel_width) ** 2)
    super_pixler = SuperPixler(img, super_pixel_width=super_pixel_width)

    super_pixel_model = torch.nn.Sequential(
        super_pixler,
        numpy2torch_converter,
        model,
        scoring
    )

    background_super_pixel = np.array([[1 for _ in range(n_super_pixel)]])
    image_super_pixel = np.array([[0 for _ in range(n_super_pixel)]])
    kernel_explainer = shap.KernelExplainer(super_pixel_model, background_super_pixel)

    # Very large values for nsamples cause OOM errors depending on image and super pixel parameter. We combine batches of SHAP values to distribute the load.
    collected_shap_values = np.zeros_like(background_super_pixel)

    # take shap value with highest abs. value for each pixel from each batch
    b = 10
    for i in range(b):
        print(f"{target_index}> {i / b:0.2%}")

        shap_values = kernel_explainer.shap_values(image_super_pixel, nsamples=1000)
        stacked_values = np.vstack([shap_values, collected_shap_values])
        index_max_values = np.argmax(np.abs(stacked_values), axis=0)
        collected_shap_values = stacked_values[index_max_values, range(shap_values.shape[1])]
    clear_output()
    print((collected_shap_values != 0).sum(), "non-zero shap values found")
    # plot the found SHAP values. Expected value does not match due to merging of batches
    shap.initjs()
    shap.force_plot(kernel_explainer.expected_value, collected_shap_values, show=False, matplotlib=True).savefig(
        '/content/feature' + str(target_index) + '.png')
    # match super pixels back to image pixels
    shap_to_pixel = collected_shap_values.reshape(img_size // super_pixel_width,
                                                  img_size // super_pixel_width)  # reshape to square
    shap_to_pixel = np.repeat(shap_to_pixel, super_pixel_width, axis=0)  # extend superpixles to the right
    shap_to_pixel = np.repeat(shap_to_pixel, super_pixel_width, axis=1)  # and down
    shap_to_pixel = shap_to_pixel / (
                np.max(np.abs(collected_shap_values)) * 2) + 0.5  # center values between 0 and 1 for the colour map

    return shap_to_pixel



import time

def over_all(path, img_size, confidence, iou, super_pixel_width, show_image_processing=0, show_yolo_result=0):
    start = time.time()
    img_org, img_pre, img_gray = image_processing(path, img_size, show_image_processing)
    output, prediction = model_processing(img_pre, confidence, iou, show_yolo_result)
    fig, ax = plt.subplots(1, output[0].shape[0], figsize=(output[0].shape[0] * 10, output[0].shape[0] * 10),
                           squeeze=False)

    target_index = 8
    target = output[0].cpu().numpy()[target_index, :]  # select a target from the output

    for target_index in range(output[0].shape[0]):
        target = output[0].cpu().numpy()[target_index, :]
        scoring = OD2Score(target, conf_thresh=confidence, iou_thresh=iou)
        numpy2torch_converter = CastNumpy()
        shap_to_pixels = shap_result(img_pre, img_gray, target, target_index, super_pixel_width, img_size, scoring)
        # plot image and shap values for super pixels on top
        ax[0][target_index].set_title("Super pixel contribution to target detection")
        ax[0][target_index].imshow(img_gray, cmap="gray", alpha=0.8)
        ax[0][target_index].imshow(shap_to_pixels, cmap=plt.cm.seismic, vmin=0, vmax=1, alpha=0.2)

        # Add bounding box of target
        label = f"{model.names[int(target[5])]}"
        ax[0][target_index].text(target[0], target[1], label, color="green")
        bbox = patches.Rectangle(target[:2], target[2] - target[0], target[3] - target[1], linewidth=1, edgecolor='g',
                                 facecolor='none')
        ax[0][target_index].add_patch(bbox)
        end = time.time()
        print(f'Time taken: {end-start}')


over_all('D:/Repos/yolov8_XAI/Yolov8_XAI_experiment/data/gtFine/images/test/bonn/bonn_000035_000019_leftImg8bit.png',320,0.3,0.3,16,0,1)