"""
visualize_pca.py

This script performs PCA on the DINO ViT features and visualize.
"""

from networkx import configuration_model
from transformers import ViTImageProcessor, ViTModel, ViTConfig
from PIL import Image
import requests
import torch.nn as nn
from sklearn.decomposition import PCA
import numpy as np
import cv2
import torch
import argparse
import torchvision

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


parser = argparse.ArgumentParser(
    description="Visualize PCA of features from the DINO ViT")


parser.add_argument("--url", type=str, default="https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png",
                    help="url of the image to load")
parser.add_argument("--image", type=str, default=None,
                    help="path to a local image to be loaded")
parser.add_argument("--layer", type=int, default=11,
                    help="0 indexed attention layer number")
parser.add_argument("--facet", type=str, default="key",
                    help="one of key, query and value", choices=["key", "query", "value"])
parser.add_argument("--resize-size", type=int, default=224,
                    help="size to resize the image to, must be a multiple of 16")
parser.add_argument("--invert-pca", action="store_true",
                    help="invert the pca features")
parser.add_argument("--remove-background",
                    action="store_true", help="remove the background")

args = parser.parse_args()

if args.image:
    image = Image.open(args.image)
else:
    url = args.url
    image = Image.open(requests.get(url, stream=True).raw)

# Load the pretrained dino model
resize_size = args.resize_size
assert resize_size == 224, "only 224 is supported for now"
assert resize_size % 16 == 0, "resize size must be a multiple of 16"
n_patches_side = resize_size // 16

processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
torch_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((resize_size, resize_size)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])
config = ViTConfig.from_pretrained('facebook/dino-vitb16')
model = ViTModel.from_pretrained('facebook/dino-vitb16').cuda()
# print(model.__dir__())
# print(model.config)
# model.config["image_size"] = resize_size
# exit()

features = None  # This global variable stores the patches

# We use this hook to get intermediate features
def get_features_hook(module: nn.Module, inp, out):
    global features
    # We don't the token corresponding to the [CLS] token
    features = out[0][1:]


layer = args.layer
facet = args.facet
# Attach hook to the given of ViT encoder
if facet == "key":
    model.encoder.layer[layer].attention.attention.key.register_forward_hook(
        get_features_hook)
elif facet == "query":
    model.encoder.layer[layer].attention.attention.query.register_forward_hook(
        get_features_hook)
elif facet == "value":
    model.encoder.layer[layer].attention.attention.value.register_forward_hook(
        get_features_hook)
else:
    raise NotImplementedError(f"facet {facet} is not implemented")

# Pass input
# inputs = processor(images=image, return_tensors="pt")
# print(inputs.keys())
# exit()
with torch.no_grad():
    outputs = model(pixel_values=torch_transform(image)[None].to('cuda'))

# Convert features to numpy array for pca analysis
features = features.cpu().numpy()


pca = PCA(n_components=3)
pca_out = pca.fit_transform(features)
# Convert flattened pca features to back to image
pca_out = pca_out.reshape((n_patches_side, n_patches_side, 3))
if args.invert_pca:
    pca_out = -pca_out
if args.remove_background:
    pca_out[pca_out[:, :, 0] < 0] = [0, 0, 0]
    pca_out[pca_out < 0] = 0
pca_out[:, :, 0] = (pca_out[..., 0] - pca_out[..., 0].min()) / \
    (pca_out[..., 0].max()-pca_out[..., 0].min())
pca_out[:, :, 1] = (pca_out[..., 1] - pca_out[..., 1].min()) / \
    (pca_out[..., 1].max()-pca_out[..., 1].min())
pca_out[:, :, 2] = (pca_out[..., 2] - pca_out[..., 2].min()) / \
    (pca_out[..., 2].max()-pca_out[..., 2].min())

# pca_out = (pca_out - pca_out.min())/(pca_out.max()-pca_out.min())


img_in = np.asarray(image)

pca_out = (np.clip(cv2.resize(
    pca_out, img_in.shape[:-1][::-1],interpolation=cv2.INTER_NEAREST), 0, 1.0)*255).astype(np.uint8)
img_out = cv2.addWeighted(img_in, 0.5, pca_out, 0.5, 1)

cv2.namedWindow("input", cv2.WINDOW_NORMAL)
cv2.namedWindow(f"pca layer: {layer} facet: {facet}", cv2.WINDOW_NORMAL)
cv2.namedWindow("superimposed output", cv2.WINDOW_NORMAL)

cv2.imshow("input", img_in[:, :, ::-1])
cv2.imshow(f"pca layer: {layer} facet: {facet}", pca_out[:, :, ::-1])
cv2.imshow("superimposed output", img_out[:, :, ::-1])
cv2.waitKey()

# Following urls are for easy copy-paste
# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# url = 'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png'
