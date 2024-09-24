import matplotlib.pylab as pl
import numpy as np
import shap
from PIL import Image
from skimage.segmentation import slic
from torchvision.transforms import transforms

from ultralytics import YOLO

feature_names = ['small_vehicle', 'person', 'large_vehicle', 'two-wheeler', 'On-rails']


# define a function that depends on a binary mask representing if an image region is hidden
def mask_image(zs, segmentation, image, background=None):
    if background is None:
        background = image.mean((0, 1))
    out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
    for i in range(zs.shape[0]):
        out[i, :, :, :] = image
        for j in range(zs.shape[1]):
            if zs[i, j] == 0:
                out[i][segmentation == j, :] = background
    return out


def f(z):
    return model.predict(mask_image(z, segments_slic, img_orig, 255))[0]


model = YOLO('./models/best-14870.pt')  # Load the YOLOv8m-seg model

# load an image
file = "./data/gtFine/images/test/bielefeld/bielefeld_000000_000321_leftImg8bit.png"

# Load and preprocess the image
img = Image.open(file).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])
img_orig = transform(img).unsqueeze(0)  # Add batch dimension

# segment the image, so we don't have to explain every pixel
segments_slic = slic(img_orig, n_segments=50, compactness=30, sigma=3)

# use Kernel SHAP to explain the network's predictions
explainer = shap.KernelExplainer(f, np.zeros((1, 50)))
shap_values = explainer.shap_values(np.ones((1, 50)), nsamples=1000)

# get the top predictions from the model
preds = model.predict(np.expand_dims(img_orig.copy(), axis=0))
top_preds = np.argsort(-preds[0].numpy())
# Load the YOLO model


# make a color map
from matplotlib.colors import LinearSegmentedColormap

colors = []
for l in np.linspace(1, 0, 100):
    colors.append((245 / 255, 39 / 255, 87 / 255, l))
for l in np.linspace(0, 1, 100):
    colors.append((24 / 255, 196 / 255, 93 / 255, l))
cm = LinearSegmentedColormap.from_list("shap", colors)


def fill_segmentation(values, segmentation):
    out = np.zeros(segmentation.shape)
    for i in range(len(values)):
        out[segmentation == i] = values[i]
    return out


# plot our explanations
fig, axes = pl.subplots(nrows=1, ncols=4, figsize=(12, 4))
inds = top_preds[0]
axes[0].imshow(img)
axes[0].axis('off')
max_val = np.max([np.max(np.abs(shap_values[i][:, :-1])) for i in range(len(shap_values))])
for i in range(3):
    m = fill_segmentation(shap_values[inds[i]][0], segments_slic)
    axes[i + 1].set_title(feature_names[inds[i]])
    axes[i + 1].imshow(img.convert('LA'), alpha=0.15)
    im = axes[i + 1].imshow(m, cmap=cm, vmin=-max_val, vmax=max_val)
    axes[i + 1].axis('off')
cb = fig.colorbar(im, ax=axes.ravel().tolist(), label="SHAP value", orientation="horizontal", aspect=60)
cb.outline.set_visible(False)
pl.imsave("shap_explanation.png", fig)
