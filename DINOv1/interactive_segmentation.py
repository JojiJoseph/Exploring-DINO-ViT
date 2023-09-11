"""
interactive_segmentation.py

This script performs segmentation using the DINO ViT features and visualize.

Usage:
interactive_segmentation.py --url <url> [--layer <layer> --facet <key|query|value>] --thresh_mode [0,1]
interactive_segmentation.py --image <path/to/image> [--layer <layer> --facet <key|query|value>] --thresh_mode [0,1]
"""

from collections import deque
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
interactive_segmentation.py --url <url> [--layer <layer> --facet <key|query|value>] --thresh_mode [0,1]
interactive_segmentation.py --image <path/to/image> [--layer <layer> --facet <key|query|value>] --thresh_mode [0,1]

Click anyware on the image to segment.

If both path to local image and url are provided, image will be loaded from local
"""
parser = argparse.ArgumentParser(
    description="interactive_segmentation using features from the DINO ViT", usage=usage)


parser.add_argument("--url", type=str, default="https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png",
                    help="url of the image to load")
parser.add_argument("--image", type=str, default=None,
                    help="path to a local image to be loaded")
parser.add_argument("--layer", type=int, default=11,
                    help="0 indexed attention layer number")
parser.add_argument("--facet", type=str, default="key",
                    help="one of key, query and value")
parser.add_argument("--thresh_mode", type=int, default=0,
                    help="Difficult to explain. See code :)")

args = parser.parse_args()
thresh_mode = args.thresh_mode

if args.image:
    image = Image.open(args.image)
else:
    url = args.url
    image = Image.open(requests.get(url, stream=True).raw)

image = image.resize((224, 224))

# Load the pretrained dino model
processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
model = ViTModel.from_pretrained('facebook/dino-vitb16')

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
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Convert features to numpy array for pca analysis
features = features.cpu().numpy()


features = np.reshape(features, (14, 14, -1))

img_in = np.asarray(image)

cv2.namedWindow("input", cv2.WINDOW_NORMAL)
cv2.namedWindow("output", cv2.WINDOW_NORMAL)
cv2.createTrackbar("threshold", "input", 50, 100, lambda x: x)


def on_mouse_click(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONUP:
        return
    x = x//16
    y = y//16
    current_feature = features[y][x]
    mask = np.zeros((14, 14)).astype(np.uint8)
    mask[y][x] = 255
    visited = set([y, x])
    q = deque([(y, x)])
    threshold = cv2.getTrackbarPos("threshold", "input")/100
    while q:
        y, x = q.popleft()
        if thresh_mode == 1:
            current_feature = features[y][x]
        for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (1, -1), (1, 0), (1, 1), (0, -1), (0, 1)]:
            if 0 <= y+dy < 14 and 0 <= x+dx < 14 and (y+dy, x+dx) not in visited:
                visited.add((y+dy, x+dx))
                if np.dot(current_feature, features[y+dy][x+dx])/np.linalg.norm(current_feature)/np.linalg.norm(features[y+dy][x+dx]) > threshold:
                    mask[y+dy][x+dx] = 255
                    q.append((y+dy, x+dx))
    mask = cv2.resize(mask, (224, 224))
    cv2.imshow("output", np.bitwise_and(img_in, mask[:, :, None])[:, :, ::-1])


cv2.setMouseCallback("input", on_mouse_click)
while True:

    cv2.imshow("input", img_in[:, :, ::-1])
    key = cv2.waitKey(10) & 0xFF
    if key in [ord('q'), 27]:
        break

# Following urls are for easy copy-paste
# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# url = 'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png'
