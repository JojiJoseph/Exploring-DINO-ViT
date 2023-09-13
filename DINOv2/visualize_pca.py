"""
visualize_pca.py

This script performs PCA on the DINOv2 features and visualize.
"""

from PIL import Image
import requests
from scipy.datasets import face
import torch.nn as nn
from sklearn.decomposition import PCA
import numpy as np
import cv2
import torch
import torchvision
import argparse

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
                    help="one of key, query and value", choices=["key", "query", "value", "all"])
parser.add_argument("--resize-size", type=int, default=518,
                    help="size to resize the image to, must be a multiple of 14")
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
dino_model = torch.hub.load(
    'facebookresearch/dinov2', 'dinov2_vitb14').to('cuda')
layer = args.layer
facet = args.facet

resize_size = args.resize_size

assert resize_size % 14 == 0, "resize size must be a multiple of 14"
n_patches_side = resize_size // 14


torch_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((resize_size, resize_size)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])
image_torch = torch_transform(image)[None].to('cuda')


features = None  # This global variable stores the patches

# We use this hook to get intermediate features


def get_features_hook(module: nn.Module, inp, out):
    global features
    # We don't the token corresponding to the [CLS] token
    features = out[0][1:]


dino_model.blocks[layer].attn.qkv.register_forward_hook(
    get_features_hook)

with torch.no_grad():
    dino_model(image_torch)

features = features.cpu().numpy()

feature_dim = features.shape[-1] // 3
if facet == "key":
    features = features[:, feature_dim:2*feature_dim]
elif facet == "query":
    features = features[:, :feature_dim]
elif facet == "value":
    features = features[:, 2*feature_dim:]
elif facet == "all":
    pass  # Do nothing. Only for readability
else:
    raise NotImplementedError(f"facet {facet} is not implemented")


pca = PCA(n_components=3)
pca_out = pca.fit_transform(features)

pca_out = pca_out.reshape((n_patches_side, n_patches_side, 3))

if args.invert_pca:
    pca_out = -pca_out
if args.remove_background:
    pca_out[pca_out[:, :, 0] < 0] = [0, 0, 0]
    pca_out[pca_out < 0] = 0
    pca_out_flatten = pca_out[:, :, 0].flatten()
    pca_out = pca_out.reshape((-1, 3))
    featuers_survived = features[pca_out_flatten > 0]
    pca_new = PCA(n_components=3)
    pca_out_new = pca_new.fit_transform(featuers_survived)
    # print(pca_out.shape, pca_out_new.shape, pca_out_flatten.shape)
    pca_out[pca_out_flatten > 0] = pca_out_new
    pca_out[pca_out_flatten > 0] -= pca_out[pca_out_flatten > 0].min(axis=0)
    pca_out = pca_out.reshape((n_patches_side, n_patches_side, 3))
pca_out[:, :, 0] = (pca_out[..., 0] - pca_out[..., 0].min()) / \
    (pca_out[..., 0].max()-pca_out[..., 0].min())
pca_out[:, :, 1] = (pca_out[..., 1] - pca_out[..., 1].min()) / \
    (pca_out[..., 1].max()-pca_out[..., 1].min())
pca_out[:, :, 2] = (pca_out[..., 2] - pca_out[..., 2].min()) / \
    (pca_out[..., 2].max()-pca_out[..., 2].min())

img_in = np.asarray(image)

pca_out = (np.clip(cv2.resize(
    pca_out, img_in.shape[:-1][::-1], interpolation=cv2.INTER_NEAREST), 0, 1.0)*255).astype(np.uint8)
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
