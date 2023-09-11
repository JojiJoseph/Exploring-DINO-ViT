"""
visualize_pca.py

This script performs PCA on the DINOv2 features and visualize.
"""

from PIL import Image
import requests
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
                    help="one of key, query and value", choices=["key", "query", "value"])

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



torch_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])
image_torch = torch_transform(image)[None].to('cuda')


features = None  # This global variable stores the patches

# We use this hook to get intermediate features


def get_features_hook(module: nn.Module, inp, out):
    global features
    print(out.shape)
    # We don't the token corresponding to the [CLS] token
    features = out[0][1:]


dino_model.blocks[layer].attn.qkv.register_forward_hook(
    get_features_hook)

with torch.no_grad():
    dino_model(image_torch)

features = features.cpu().numpy()

if facet == "key":
    features = features[:, :384]
elif facet == "query":
    features = features[:, 384:768]
elif facet == "value":
    features = features[:, 768:]
else:
    raise NotImplementedError(f"facet {facet} is not implemented")


pca = PCA(n_components=3)
pca_out = pca.fit_transform(features)
# Convert flattened pca features to back to image
pca_out = pca_out.reshape((16, 16, 3))
pca_out = (pca_out - pca_out.min())/(pca_out.max()-pca_out.min())


img_in = np.asarray(image)

pca_out = (np.clip(cv2.resize(
    pca_out, img_in.shape[:-1][::-1]), 0, 1.0)*255).astype(np.uint8)
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
