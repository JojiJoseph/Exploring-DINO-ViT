from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests
import torch.nn as nn
from sklearn.decomposition import PCA
import numpy as np
import cv2

url1 = 'https://raw.githubusercontent.com/ShirAmir/dino-vit-features/main/images/cat.jpg'
url2 = 'https://raw.githubusercontent.com/ShirAmir/dino-vit-features/main/images/ibex.jpg'

image1 = Image.open(requests.get(url1, stream=True).raw)
image2 = Image.open(requests.get(url2, stream=True).raw)


# Load the pretrained dino model
processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
model = ViTModel.from_pretrained('facebook/dino-vitb16')

features = None # This global variable stores the patches

# We use this hook to get intermediate features
def get_features_hook(module: nn.Module, inp, out):
    global features
    features = out[0][1:] # We don't the token corresponding to the [CLS] token
# Attach hook to last layer of ViT encoder
handle = model.encoder.layer[-2].attention.attention.value.register_forward_hook(get_features_hook)

# Pass input
inputs = processor(images=image1, return_tensors="pt")
outputs = model(**inputs)

# Convert features to numpy array
features1 = features.detach().cpu().numpy()
handle.remove()

# Attach hook to last layer of ViT encoder
handle = model.encoder.layer[-2].attention.attention.value.register_forward_hook(get_features_hook)

# Pass input
inputs = processor(images=image2, return_tensors="pt")
outputs = model(**inputs)

# Convert features to numpy array
features2 = features.detach().cpu().numpy()
handle.remove()

points = []

cv2.namedWindow("Select some points and press space")
xind, yind = 0, 0 # x and y indicator
image1 = np.asarray(image1)
image1 = cv2.resize(image1, (640, 480))
sh, sw, _ = image1.shape
def on_mouse(event,x,y,flags,param):
    global xind, yind
    xind = (x/sw)*224/16
    xind = int(int(xind) * 16 * sw/224) 
    yind = (y/sh)*224/16
    yind = int(int(yind) * 16 * sh/224)
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([xind, yind])
    
cv2.setMouseCallback("Select some points and press space", on_mouse)



while True:
    img_with_points = image1.copy()
    for point in points:
        cv2.circle(img_with_points, point, 5, (0, 0, 255), -1)
    print(xind, yind)
    cv2.circle(img_with_points, (xind,yind), 5, (0, 255, 0), -1)
    cv2.imshow("Select some points and press space", img_with_points)
    key = cv2.waitKey(10) & 0xFF
    # print(key)
    if key == ord(' '):
        break

src_point_idx = []
dest_point_idx = []
sh, sw, _ = image1.shape
image2 = np.asarray(image2)

image1_rescaled = cv2.resize(image1, (224, 224))
image2_rescaled = cv2.resize(image2, (224, 224))
img_stack = np.hstack([image1_rescaled, image2_rescaled])
print(img_stack.shape)
for point in points:
    x, y = point
    x = x * 14/sw
    y = y * 14/sh
    x = int(x)
    y = int(y)
    sidx = 14 * y + x
    # print(np.dot(features2,features1[sidx]).shape,np.sqrt(np.sum(features2**2,axis=1)).shape,np.sqrt(np.sum(features1[sidx]**2)).shape)
    # exit()
    didx = np.argmax(np.dot(features2,features1[sidx])/np.sqrt(np.sum(features2**2,axis=1))/np.sqrt(np.sum(features1[sidx]**2)))
    dy = didx//14
    dx = didx % 14
    # print(np.int0((x * 16, y*16)).dtype,np.int0((224 + dx*16, dy*16)).dtype)
    # print(x, y, dy, dx)
    # print([int(x * 16), int(y*16)], [int(224 + dx*16), int(dy*16)])
    cv2.line(img_stack,[int(x * 16), int(y*16)], [int(224 + dx*16), int(dy*16)], (0, 255,0),2)
cv2.imshow("image stack", img_stack)
cv2.waitKey()
