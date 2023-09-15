from collections import defaultdict
from copy import deepcopy
from itertools import product
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel
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
                    help="one of key, query and value", choices=["key", "query", "value", "all"])
parser.add_argument("--resize-size", type=int, default=518,
                    help="size to resize the image to, must be a multiple of 14")
parser.add_argument("--invert-pca", action="store_true",
                    help="invert the pca features")
parser.add_argument("--remove-background",
                    action="store_true", help="remove the background")

args = parser.parse_args()

facet = args.facet


url1 = 'https://raw.githubusercontent.com/ShirAmir/dino-vit-features/main/images/cat.jpg'
url2 = 'https://raw.githubusercontent.com/ShirAmir/dino-vit-features/main/images/ibex.jpg'

# image1 = Image.open(requests.get(url1, stream=True).raw)
# image2 = Image.open(requests.get(url2, stream=True).raw)

image1 = Image.open("../figures/cat2.jpg")
image2 = Image.open("../figures/cat1.jpg")

# image1 = Image.open("../figures/chimp.jpeg")
# image2 = Image.open("../figures/lena_test_image.png")

# image2 = Image.open("./Bocksbeutel_bottle.jpg")
# image1 = Image.open("./Botella_de_plástico_-_PET.jpg")
# image1 = Image.open("../figures/car1.jpg")
# image2 = Image.open("../figures/car2.jpg")

# Load the pretrained dino model
dino_model = torch.hub.load(
    'facebookresearch/dinov2', 'dinov2_vitl14').to('cuda')
resize_size  = args.resize_size
n_side = resize_size // 14
torch_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((resize_size, resize_size)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])


features = None # This global variable stores the patches

# We use this hook to get intermediate features
def get_features_hook(module: nn.Module, inp, out):
    global features
    features = out[0][1:,:] # We don't the token corresponding to the [CLS] token

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
feature_dim = features1.shape[-1] // 3
if facet == "key":
    features1 = features1[:, feature_dim:2*feature_dim]
    features2 = features2[:, feature_dim:2*feature_dim]
elif facet == "query":
    features1 = features1[:, :feature_dim]
    features2 = features2[:, :feature_dim]
elif facet == "value":
    features1 = features1[:, 2*feature_dim:]
    features2 = features2[:, 2*feature_dim:]
elif facet == "all":
    pass  # Do nothing. Only for readability
else:
    raise NotImplementedError(f"facet {facet} is not implemented")

points = []

cv2.namedWindow("Select some segments and press space", cv2.WINDOW_NORMAL)
xind, yind = 0, 0 # x and y indicator
image1 = np.asarray(image1)
image1 = cv2.resize(image1, (resize_size, resize_size))
sh, sw, _ = image1.shape

mouse_pressed = False
global_mask = np.zeros((sh, sw), dtype=np.uint8)
colors = [None,
          
          [0, 0, 255],
          [0, 255, 0],
          [255, 0, 0],
          [255, 255, 0],
          [255, 0, 255],
          [0, 255, 255],
          [255, 255, 255],
          [0, 0, 0],
          [128, 128, 128],]
segmentation_sets = []
set_of_patch = defaultdict(int)

global_input_mask = np.zeros((n_side, n_side, 3), dtype=np.uint8)


def on_mouse(event,x,y,flags,param):
    global xind, yind, mouse_pressed, segmentation_sets, colors, global_input_mask
    xind = (x/sw)*n_side
    xind = int(int(xind) * sw/n_side) 
    yind = (y/sh)*n_side
    yind = int(int(yind) * sh/n_side)
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        segmentation_sets.append([])
        # colors.append(np.random.randint(0, 255, 3).tolist())
        if [xind, yind] not in points:
            points.append([xind, yind])
            segmentation_sets[-1].append([int(xind/sw*n_side), int(yind/sh*n_side)])
            global_input_mask[int(yind/sh*n_side), int(xind/sw*n_side)] = colors[len(segmentation_sets)]
    if event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            if [xind, yind] not in points:
                points.append([xind, yind])
                segmentation_sets[-1].append([int(xind/sw*n_side), int(yind/sh*n_side)])
                global_input_mask[int(yind/sh*n_side), int(xind/sw*n_side)] = colors[len(segmentation_sets)]
    if event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False
    

cv2.setMouseCallback("Select some segments and press space", on_mouse)

while True:
    # cv2.imshow("global_mask", global_input_mask)
    img_with_points = image1.copy()
    mask = cv2.resize(global_input_mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
    img_with_points[mask >0 ] = img_with_points[mask>0] // 4
    img_with_points = cv2.addWeighted(img_with_points, 1, mask, 1, 1)
    for current_set, color in zip(segmentation_sets, colors[1:]):
        for point in deepcopy(current_set):
            
            print(sw, n_side, point)
            point[0] = round(point[0]*sw/n_side)
            point[1] = round(point[1]*sh/n_side)
            print(point)
            # cv2.circle(img_with_points, [point[0]+sw//(n_side*2),point[1]+sh//(n_side*2)], 5, color, -1)
            # cv2.rectangle(img_with_points, (point[0], point[1]), (point[0]+sw//n_side, point[1]+sh//n_side), color, 2)
    print(xind, yind)
    # cv2.circle(img_with_points, (xind+sw//(n_side*2),yind+sh//(n_side*2)), 5, (0, 255, 0), -1)
    cv2.rectangle(img_with_points, (xind, yind), (xind+sw//n_side, yind+sh//n_side), (0, 0, 0), 2)
    cv2.imshow("Select some segments and press space", img_with_points[...,::-1])
    key = cv2.waitKey(10) & 0xFF

    if key == ord(' '):
        break

cv2.destroyAllWindows()
src_point_idx = []
dest_point_idx = []
sh, sw, _ = image1.shape
image2 = np.asarray(image2)

image1_rescaled = cv2.resize(image1, (resize_size, resize_size))
image2_rescaled = cv2.resize(image2, (resize_size, resize_size))

for point in points:
    point[0] = int(point[0] * n_side/sw)
    point[1] = int(point[1] * n_side/sh)

background_points = []

for i,j in product(range(n_side), range(n_side)):
    if [i, j] not in points:
        background_points.append([i, j])

segmentation_sets = [background_points] + segmentation_sets

features = np.concatenate([features1, features2], axis=0)

pca = PCA(n_components=128)
pca.fit(features)
# features1 = pca.transform(features1)
# features2 = pca.transform(features2)

# print(points)

# exit()
for dest_patch in tqdm(range(n_side*n_side)):
    # print(features1.shape)
    # print(dest_patch)
    similarity = -np.inf
    for s_idx in range(len(segmentation_sets)):
        for src_patch in segmentation_sets[s_idx]:
            # print(src_patch)
            if np.dot(features1[src_patch[1]*n_side + src_patch[0]], features2[dest_patch])/np.sqrt(np.sum(features1[src_patch[1]*n_side + src_patch[0]]**2))/np.sqrt(np.sum(features2[dest_patch]**2)) > similarity:
                similarity = np.dot(features1[src_patch[1]*n_side + src_patch[0]], features2[dest_patch])/np.sqrt(np.sum(features1[src_patch[1]*n_side + src_patch[0]]**2))/np.sqrt(np.sum(features2[dest_patch]**2))
                if similarity > 0.0:
                    set_of_patch[dest_patch] = s_idx

print(set(set_of_patch.values()))
print(np.array(segmentation_sets).shape)
# exit()
mask_in = cv2.resize(global_input_mask, (resize_size, resize_size), interpolation=cv2.INTER_NEAREST)
image1_rescaled[mask_in > 0] = image1_rescaled[mask_in > 0]//4
image1_rescaled = cv2.addWeighted(image1_rescaled, 1, mask_in, 1, 1)
# print(set_of_patch)
mask_out = np.zeros((n_side, n_side, 3), dtype=np.uint8)
mask_out_grabcut = np.zeros((n_side, n_side, len(set(set_of_patch.values()))), dtype=np.uint8)
for dest_patch in tqdm(range(n_side*n_side)):
    if set_of_patch[dest_patch] != 0:
        color = colors[set_of_patch[dest_patch]]
        x = dest_patch % n_side
        y = dest_patch // n_side
        mask_out_grabcut[y, x, set_of_patch[dest_patch]] = 255
        mask_out[y,x] = color
        # cv2.rectangle(img_stack, (resize_size + x*14 + 7, y*14 + 7), (resize_size + x*14 + 7 + 14, y*14 + 7 + 14), color, 2)
def get_grab_cut_mask(image, mask, colors):
    resize_size = image.shape[0]
    mask = cv2.resize(mask, (resize_size, resize_size), interpolation=cv2.INTER_NEAREST)
    output_mask = np.zeros_like(image, dtype=np.uint8)
    for i in range(1, mask.shape[-1]):
        current_mask = mask[:,:,i].copy()
        pr_fgd_indices = current_mask > 0
        current_mask = np.ascontiguousarray(current_mask)
        forground_mask = cv2.erode(current_mask, np.ones((7,7), dtype=np.uint8), iterations=2)
        background_mask = cv2.erode(255-current_mask, np.ones((7,7), dtype=np.uint8), iterations=1)
        fg_indices = (forground_mask == 255)
        bg_indices = (background_mask == 255)
        # current_mask[...] = cv2.GC_BGD
        current_mask[...] = cv2.GC_PR_FGD
        # current_mask[pr_fgd_indices] = cv2.GC_PR_FGD
        current_mask[bg_indices] = cv2.GC_BGD
        current_mask[fg_indices] = cv2.GC_FGD
        print(cv2.GC_PR_BGD, cv2.GC_PR_FGD, cv2.GC_BGD, cv2.GC_FGD)
        # exit()
        temp_mask = current_mask.copy()
        temp_mask[temp_mask==0] = 5
        temp_mask[bg_indices] = cv2.GC_BGD
        cv2.imshow("current_mask", temp_mask*50)
        cv2.waitKey()

        try:
            bgdModel = np.zeros((1,65),np.float64)
            fgdModel = np.zeros((1,65),np.float64)
            current_mask = np.ascontiguousarray(current_mask)
            cv2.grabCut(image, current_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        except:
            pass
        print(np.unique(current_mask))
        current_mask = np.where((current_mask==2)|(current_mask==0),0,1).astype('uint8')
        current_mask = np.ascontiguousarray(current_mask, dtype=np.uint8)
        output_mask[current_mask>0]  = colors[i]
        # print(current_mask.shape)
        # print(np.unique(current_mask))
    return output_mask
mask_out_grabcut = get_grab_cut_mask(image2_rescaled, mask_out_grabcut, colors)
print(mask_out.shape)
cv2.imshow("mask_out", mask_out)
cv2.waitKey()
# exit()
for current_set, color in zip(segmentation_sets, colors):
    
    if color is None:
        continue
    # print(set)
    for point in current_set:
        x, y = point
        x = round(x)
        y = round(y)
        # mask_out[y, x] = color
    #     cv2.rectangle(img_stack, (x*14 + 7, y*14 + 7), (x*14 + 7 + 14, y*14 + 7 + 14), color, 2)
# exit()
mask_out = cv2.resize(mask_out, (resize_size, resize_size), interpolation=cv2.INTER_NEAREST)
# image2_copy = image2_rescaled.copy()
image3_rescaled = image2_rescaled.copy()
image2_rescaled[mask_out > 0] = image2_rescaled[mask_out > 0]//4
image2_rescaled = cv2.addWeighted(image2_rescaled, 1, mask_out, 1, 1)
image3_rescaled = cv2.addWeighted(image3_rescaled, 1, mask_out_grabcut, 1, 1)

# print(np.unique(mask_out), mask_out.shape)
# mask_out = np.sum(mask_out, axis=-1)
# pr_fgd_indices = mask_out > 0
# mask_out[mask_out>0] = 255
# mask_out = np.ascontiguousarray(mask_out, dtype=np.uint8)
# cv2.erode(mask_out, np.ones((7,7), dtype=np.uint8), iterations=1, dst=mask_out)
# print(np.unique(mask_out))
# fg_indices = (mask_out == 255)
# mask_out = np.zeros_like(mask_out, dtype=np.uint8)
# mask_out[...] = cv2.GC_BGD
# mask_out[pr_fgd_indices] = cv2.GC_PR_FGD
# mask_out[fg_indices] = cv2.GC_FGD
# print(cv2.GC_BGD, cv2.GC_PR_BGD, cv2.GC_BGD)
# # mask_out = mask_out[..., 0]
# # print(mask_out.shape)



# bgdModel = np.zeros((1,65),np.float64)
# fgdModel = np.zeros((1,65),np.float64)
# print(mask_out.shape, mask_out.dtype)
# cv2.grabCut(image2_copy, np.ascontiguousarray(mask_out), None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
# mask2 = np.where((mask_out==2)|(mask_out==0),0,1).astype('uint8')
# print(mask2.shape)
# image2_new = image2_copy*mask2[:,:,np.newaxis]
# print(image2_new.shape, image2_new.dtype)
# # exit()
# cv2.imshow("grabcut", image2_new[...,::-1])
# cv2.waitKey()

img_stack = np.hstack([image1_rescaled, image2_rescaled, image3_rescaled])
# img_stack[resize_size + mask_in > 0] = img_stack[resize_size + mask_in > 0]//2
cv2.namedWindow("image stack", cv2.WINDOW_NORMAL)
cv2.imshow("image stack", img_stack[...,::-1])
cv2.waitKey()

cv2.imwrite("image_stack.png", img_stack[...,::-1])
