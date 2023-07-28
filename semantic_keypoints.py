from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests
import torch.nn as nn
from sklearn.decomposition import PCA
import numpy as np
import cv2

url1 = 'https://raw.githubusercontent.com/ShirAmir/dino-vit-features/main/images/cat.jpg'
url2 = 'https://raw.githubusercontent.com/ShirAmir/dino-vit-features/main/images/ibex.jpg'

# image1 = Image.open(requests.get(url1, stream=True).raw)
# image2 = Image.open(requests.get(url2, stream=True).raw)

image1 = Image.open('cat1.jpg')
image2 = Image.open('cat2.jpg')

image1 = Image.open('plane.jpeg')
image2 = Image.open('bird.jpg')

image1 = Image.open('bag1.jpg')
image2 = Image.open('bag3.jpg')

image1 = Image.open('chimp.jpeg')
image2 = Image.open('lena_test_image.png')

image1 = image1.resize((224, 224))
image2 = image2.resize((224, 224))


# Load the pretrained dino model
processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
model = ViTModel.from_pretrained('facebook/dino-vitb16')

features = None # This global variable stores the patches

# We use this hook to get intermediate features
def get_features_hook(module: nn.Module, inp, out):
    global features
    features = out[0][1:] # We don't the token corresponding to the [CLS] token
# Attach hook to last layer of ViT encoder
handle = model.encoder.layer[-2].attention.attention.key.register_forward_hook(get_features_hook)

# Pass input
inputs = processor(images=image1, return_tensors="pt")
outputs = model(**inputs)

# Convert features to numpy array
features1 = features.detach().cpu().numpy()
handle.remove()

# Attach hook to last layer of ViT encoder
handle = model.encoder.layer[-2].attention.attention.key.register_forward_hook(get_features_hook)

# Pass input
inputs = processor(images=image2, return_tensors="pt")
outputs = model(**inputs)

# Convert features to numpy array
features2 = features.detach().cpu().numpy()
handle.remove()

image1_rescaled = cv2.resize(np.asarray(image1), (224, 224))
image2_rescaled = cv2.resize(np.asarray(image2), (224, 224))
img_stack = np.hstack([image1_rescaled, image2_rescaled])


dist = [0] * 196
matchidx = [0] * 196
from scipy.special import softmax
for sidx in range(196):
    didx = np.argmax(np.dot(features2,features1[sidx])/np.sqrt(np.sum(features2**2,axis=1))/np.sqrt(np.sum(features1[sidx]**2)))
    ridx = np.argmax(np.dot(features1,features2[didx])/np.sqrt(np.sum(features1**2,axis=1))/np.sqrt(np.sum(features2[didx]**2))) # reverse index
    dist[sidx] = abs(ridx-sidx)
    if abs(ridx-sidx) == 0:
        dist[sidx] = 1-(np.argmax(np.dot(features2,features1[sidx])/np.sqrt(np.sum(features2**2,axis=1))/np.sqrt(np.sum(features1[sidx]**2)))) * (np.argmax(np.dot(features1,features2[didx])/np.sqrt(np.sum(features1**2,axis=1))/np.sqrt(np.sum(features2[didx]**2))))
        dist[sidx] = 1-(softmax(np.dot(features2,features1[sidx])/np.sqrt(np.sum(features2**2,axis=1))/np.sqrt(np.sum(features1[sidx]**2))))[didx] * (softmax(np.dot(features1,features2[didx])/np.sqrt(np.sum(features1**2,axis=1))/np.sqrt(np.sum(features2[didx]**2)))[sidx])
    else:
        dist[sidx] = 10000
    matchidx[sidx] = didx
# print(dist)
# exit()
idx_array = np.argsort(dist)

for i in range(10):
    y = idx_array[i] // 14
    x = idx_array[i] % 14
    y *= 16
    x *= 16
    # print(x, y)
    y2 = matchidx[idx_array[i]] // 14
    x2 = matchidx[idx_array[i]] % 14
    y2 *= 16
    x2 *= 16
    cv2.line(img_stack, (x+8,y+8), (224+x2+8, y2+8), (0,255,0))
    cv2.circle(img_stack, (x+8,y+8), 2, (255,0,0), -1)
    cv2.putText(img_stack,str(i),(x+8,y+8),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,128))
    # print(x2, y2)
    cv2.circle(img_stack, (224+x2+8,y2+8), 2, (255,0,0), -1)
    cv2.putText(img_stack,str(i),(224+x2+8,y2+8),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,128))

img_stack = cv2.resize(img_stack, (448*2, 448))
cv2.imshow("out", img_stack[:,:,::-1])
cv2.waitKey()