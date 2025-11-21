import os
import cv2
import numpy as np
from glob import glob
# type hints 
from typing import List, Tuple
import math


try: 
    import torch
    from torchvision import transforms
    status = True



except Exception: 
    status = False

    #end program without PyTorch 
    if not status: 
        raise ModuleNotFoundError("'error'; PyTorch not found")



# def sort_img(folder: "the", ext=".png") -> List[str]:  #change issue with folder path later 


#     #locate files in folder 
#     files = os.listdir(folder)


#     imgs = [f for f in files]


#     return [os.path.join(folder, img) for img in imgs]



def process_img(path) -> np.ndarray: 


    img = cv2.imread(path, cv2.IMREAD_COLOR) 


    if img is None: 
        raise IOError(f"'error'; Could not load image: {path}")
    
    
    else: 


        # Change color scale to RBG
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RBG)


    # Rescale cropped image to standard 112 by 112 
    crop_img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)


    return crop_img


# Normalize img to [0, 1] pixel scale then standardize by using mean and std
# Note: Mean and standard deviation based on ImageNet dataset
def normalize_img(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> np.ndarray: 


    # from 8-bit to 32-bit float and normalize to [0,1] scale
    img_float = img.astype(np.float32) / 255.0


    # Z-score nomalization for each color channel in RBG
    for c in range(3): 
        img_float[..., c] = (img_float[..., c] - mean[c]) / std[c]


def img_to_clip(img_paths): 


    imgs = []


    # Load processed images onto list 
    for n in img_paths: 

        img = process_img(n)
        img = normalize_img(img)
        imgs.append(img)


    clp_size = len(imgs)
    clips = [] 


    # Create clips of 16 frames from 64 frame output (might change to 32 from 100 depending on testing) 
    for n in range(0, clp_size - 15): 


        c = imgs[n: n+clp_size]


        # Stack image clips in matrix (adds time dimension for 3D CNN)
        arr = np.stack(c, axis=0)


        # Transpose array to (C, T, H, W) 
        # .copy() added after debugging issues - without .copy() NumPy creates non-contiguous array 
        arr = np.transpose(arr, (3, 0, 1, 2)).copy()


        # Converts clips to torch tensors and adds them to list 
        t_clip = torch.from_numpy(c)
        clips.append(t_clip) 


# Note: found optical flow as a potential addition to image processing to boost detection accuracy
# Update: choose to stick with a 3D CNN as main neural network to generate predictions so optical flow is not necessary