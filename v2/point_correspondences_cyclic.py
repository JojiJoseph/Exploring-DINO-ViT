from PIL import Image
import requests
import torch.nn as nn
from sklearn.decomposition import PCA
import numpy as np
import cv2
import torch
import torchvision
from scipy.special import softmax

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

url1 = 'https://raw.githubusercontent.com/ShirAmir/dino-vit-features/main/images/cat.jpg'
url2 = 'https://raw.githubusercontent.com/ShirAmir/dino-vit-features/main/images/ibex.jpg'

image1 = Image.open(requests.get(url1, stream=True).raw)
image2 = Image.open(requests.get(url2, stream=True).raw)

image1 = Image.open("../figures/cat1.jpg")
image2 = Image.open("../figures/cat2.jpg")


# Load the pretrained dino model
dino_model = torch.hub.load(
    'facebookresearch/dinov2', 'dinov2_vitb14').to('cuda')
torch_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])


features = None # This global variable stores the patches

# We use this hook to get intermediate features
def get_features_hook(module: nn.Module, inp, out):
    global features
    features = out[0][1:,384:768] # We don't the token corresponding to the [CLS] token

# Attach hook to last layer of DINOv2
handle = dino_model.blocks[-1].attn.qkv.register_forward_hook(
    get_features_hook)

# Pass input
image1_torch = torch_transform(image1)[None].to('cuda')
with torch.no_grad():
    outputs = dino_model(image1_torch)

# Convert features to numpy array
features1 = features.cpu().numpy()
handle.remove()

# Attach hook to last layer of DINOv2
handle = dino_model.blocks[-1].attn.qkv.register_forward_hook(
    get_features_hook)

# Pass input
image2_torch = torch_transform(image2)[None].to('cuda')
with torch.no_grad():
    outputs = dino_model(image2_torch)

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
    xind = (x/sw)*16
    xind = int(int(xind) * sw/16) 
    yind = (y/sh)*16
    yind = int(int(yind) * sh/16)
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([xind, yind])
    
cv2.setMouseCallback("Select some points and press space", on_mouse)



while True:
    img_with_points = image1.copy()
    for point in points:
        cv2.circle(img_with_points, [point[0]+sw//32,point[1]+sh//32] , 5, (0, 0, 255), -1)
    print(xind, yind)
    cv2.circle(img_with_points, (xind+sw//32,yind+sh//32), 5, (0, 255, 0), -1)
    cv2.rectangle(img_with_points, (xind, yind), (xind+sw//16, yind+sh//16), (0, 255, 0), 2)
    cv2.imshow("Select some points and press space", img_with_points)
    key = cv2.waitKey(10) & 0xFF

    if key == ord(' '):
        break

src_point_idx = []
dest_point_idx = []
sh, sw, _ = image1.shape
image2 = np.asarray(image2)

image1_rescaled = cv2.resize(image1, (224, 224))
image2_rescaled = cv2.resize(image2, (224, 224))
img_stack = np.hstack([image1_rescaled, image2_rescaled])


for point in points:
    x, y = point
    x = x * 16/sw
    y = y * 16/sh
    x = round(x)
    y = round(y)
    sidx = 16 * y + x
    # print(np.dot(features2,features1[sidx]).shape,np.sqrt(np.sum(features2**2,axis=1)).shape,np.sqrt(np.sum(features1[sidx]**2)).shape)
    # exit()
    corresponence_scores = [] 
    for didx in range(256):
        corresponence_scores.append(
            1-(softmax(np.dot(features2,features1[sidx])/np.sqrt(np.sum(features2**2,axis=1))/np.sqrt(np.sum(features1[sidx]**2))))[didx] * (softmax(np.dot(features1,features2[didx])/np.sqrt(np.sum(features1**2,axis=1))/np.sqrt(np.sum(features2[didx]**2)))[sidx])
        )# didx = np.argmax(np.dot(features2,features1[sidx])/np.sqrt(np.sum(features2**2,axis=1))/np.sqrt(np.sum(features1[sidx]**2)))
    # ridx = 
    didx = np.argsort(corresponence_scores)[0]
    dy = didx//16
    dx = didx % 16
    # print(np.int0((x * 16, y*16)).dtype,np.int0((224 + dx*16, dy*16)).dtype)
    # print(x, y, dy, dx)
    # print([int(x * 16), int(y*16)], [int(224 + dx*16), int(dy*16)])
    cv2.line(img_stack,[int(x * 14 + 7 ), int(y*14 + 7 )], [int(224 + dx*14 + 7), int(dy*14 + 7)], (0, 255,0),2)
cv2.namedWindow("image stack", cv2.WINDOW_NORMAL)
cv2.imshow("image stack", img_stack)
cv2.waitKey()
