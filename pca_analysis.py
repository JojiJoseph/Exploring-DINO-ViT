"""
pca_analysis.py

This script performs PCA on the DINO ViT features and visualize.

Usage:
pca_analysis.py --url <url> [--layer <layer> --facet <key|query|value>]
pca_analysis.py --image <path/to/image> [--layer <layer> --facet <key|query|value>]
"""

from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests
import torch.nn as nn
from sklearn.decomposition import PCA
import numpy as np
import cv2
import torch
import argparse

usage = """
pca_analysis.py --url <url> [--layer <layer> --facet <key|query|value>]
pca_analysis.py --image <path/to/image> [--layer <layer> --facet <key|query|value>]

if both path to local image and url are provided, image will be loaded from local
"""
parser = argparse.ArgumentParser(description="Visualize PCA of features from the DINO ViT", usage=usage)


parser.add_argument("--url", type=str, default="https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png", help="url of the image to load")
parser.add_argument("--image", type=str, default=None, help="path to a local image to be loaded")
parser.add_argument("--layer", type=int, default=11, help="0 indexed attention layer number")
parser.add_argument("--facet", type=str, default="key", help="one of key, query and value")

args = parser.parse_args()

if args.image:
    image = Image.open(args.image)
else:
    url = args.url
    image = Image.open(requests.get(url, stream=True).raw)

# Load the pretrained dino model
processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
model = ViTModel.from_pretrained('facebook/dino-vitb16')

features = None  # This global variable stores the patches

# We use this hook to get intermediate features
def get_features_hook(module: nn.Module, inp, out):
    global features
    features = out[0][1:] # We don't the token corresponding to the [CLS] token


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
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Convert features to numpy array for pca analysis
features = features.cpu().numpy()


pca = PCA(n_components=3)
pca_out = pca.fit_transform(features)
# Convert flattened pca features to back to image
pca_out = pca_out.reshape((14, 14, 3))
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
