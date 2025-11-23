import cv2
import numpy as np
# type hints 
from typing import List 

# for testing purposes
#import os


try: 
    import torch
    status = True



except Exception: 
    status = False

    #end program without PyTorch. Remove later 
    if not status: 
        raise ModuleNotFoundError("'error'; PyTorch not found")





def process_img(img) -> np.ndarray: 
    

    # Change color scale to RGB 
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    # Rescale cropped image 229 by 229
    crop_img = cv2.resize(img_RGB, (299, 299), interpolation=cv2.INTER_LINEAR)


    return crop_img





# Normalize img to [0, 1] pixel scale then standardize by using mean and std
# Note: Mean and standard deviation based model input 
def normalize_img(img, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) -> np.ndarray: 


    # from 8-bit to 32-bit float and normalize to [0,1] scale
    img_float = img.astype(np.float32) / 255.0


    # Z-score nomalization for each color channel in RBG
    for c in range(3): 
        img_float[..., c] = (img_float[..., c] - mean[c]) / std[c]

    
    return img_float





# import images from detection.py
def img_to_clip(images: List[np.ndarray]) -> List[torch.Tensor]: 


    processed_imgs = []


    # Create list with processed images
    for n in images: 

        img = process_img(n)
        img = normalize_img(img)
        processed_imgs.append(img)


    clp_size = len(processed_imgs)
    clips = [] 


    # Create batch of clips (32 frames per clip - 92% overlap)
    for n in range(0, clp_size - 31, 2): 


        c = processed_imgs[n: n+31]


        # Stack image clips in matrix 
        clip_arr = np.stack(c, axis=0)


        # Transpose array to (C, T, H, W) format 
        # .copy() added after debugging issues 
        clip_arr = np.transpose(clip_arr, (3, 0, 1, 2)).copy()


        # Converts to torch tensor and add batch dimension
        tensor_clip = torch.from_numpy(clip_arr)
        batch_clip = tensor_clip.unsqueeze(axis=0) 
        clips.append(batch_clip) 



        return clips


# Note: found optical flow as a potential addition to image processing to boost detection accuracy
# Update: choose to stick with a 2D CNN as main neural network to generate predictions so optical flow is not necessary