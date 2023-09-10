"""
visualize_pca.py

This script performs PCA on the DINO ViT features and visualize.

Usage:
visualize_pca.py --url <url> [--layer <layer> --facet <key|query|value>]
visualize_pca.py --image <path/to/image> [--layer <layer> --facet <key|query|value>]
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

parser = argparse.ArgumentParser(
    description="Visualize PCA of features from the DINO ViT")


parser.add_argument("--url", type=str, default="https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png",
                    help="url of the image to load")
parser.add_argument("--image", type=str, default=None,
                    help="path to a local image to be loaded")
parser.add_argument("--layer", type=int, default=11,
                    help="0 indexed attention layer number")

args = parser.parse_args()

if args.image:
    image = Image.open(args.image)
elif args.url:
    url = args.url
    image = Image.open(requests.get(url, stream=True).raw)
else:
    raise ValueError("Either url or image must be provided")

# Load the pretrained dino model
processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
model = ViTModel.from_pretrained(
    'facebook/dino-vitb16', output_attentions=True)


attentions = None  # This global variable stores the attentions


# We use this hook to get attentions
def get_attentions_hook(module: nn.Module, inp, out):
    global attentions
    attentions = out[1][0, :, 0, 1:]


layer = args.layer

model.encoder.layer[layer].attention.attention.register_forward_hook(
    get_attentions_hook)

# Pass input
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
# exit()

# Convert attentions to numpy array for visualization
attentions = attentions.cpu().numpy()
img_in = np.asarray(image)
for idx, attention_map in enumerate(attentions):

    attention_map = attention_map.reshape((14, 14))
    attention_map = (attention_map - attention_map.min()) / \
        (attention_map.max()-attention_map.min())

    attention_map = (np.clip(cv2.resize(
        attention_map, img_in.shape[:-1][::-1]), 0, 1.0)*255).astype(np.uint8)

    img_out = cv2.addWeighted(
        img_in[:, :, ::-1], 0.25, np.repeat(attention_map[:, :, None], 3, axis=-1), 0.75, 1)
    cv2.imshow(f"Attention Map {idx}", img_out)
cv2.waitKey()

# Following urls are for easy copy-paste
# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# url = 'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png'
