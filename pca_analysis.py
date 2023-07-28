from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests
import torch.nn as nn
from sklearn.decomposition import PCA
import numpy as np
import cv2

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
url = 'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png'
image = Image.open(requests.get(url, stream=True).raw)
# image = Image.open("car1.jpg")

# Load the pretrained dino model
processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
model = ViTModel.from_pretrained('facebook/dino-vitb16')

features = None # This global variable stores the patches

# We use this hook to get intermediate features
def get_features_hook(module: nn.Module, inp, out):
    global features
    features = out[0][1:] # We don't the token corresponding to the [CLS] token

# Attach hook to last layer of ViT encoder
# print(len(model.encoder.layer))
# exit()
model.encoder.layer[6].attention.attention.key.register_forward_hook(get_features_hook)

# Pass input
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# Convert features to numpy array for pca analysis
features = features.detach().cpu().numpy()


pca = PCA(n_components=3)
pca_out = pca.fit_transform(features)
pca_out = pca_out.reshape((14,14,3)) # Convert flattened pca features to back to image
pca_out = (pca_out - pca_out.min())/(pca_out.max()-pca_out.min())


img_in = np.asarray(image)

pca_out = (np.clip(cv2.resize(pca_out, img_in.shape[:-1][::-1]),0,1.0)*255).astype(np.uint8)
img_out = cv2.addWeighted(img_in, 0.5, pca_out, 0.5, 1)

cv2.imshow("input", img_in[:,:,::-1])
cv2.imshow("pca", pca_out[:,:,::-1])
cv2.imshow("superimposed output", img_out[:,:,::-1])
cv2.waitKey()