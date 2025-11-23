import torch 
import torch.nn.functional as F
import numpy as np
from network.xception import xception
from statistics import mean

def predict(clips, batch_num): 


    # load model and weights 
    model = xception(num_classes=2, pretrained=None) 
    load_weight = torch.load("network/ffpp_c23.pth", map_location=torch.device("cpu"))
    model.load_state_dict(load_weight)


    # enable evaluation mode
    model.eval() 


    predictions = []


    # load 5D tensor and calculate frames per batch
    clip = clips[batch_num]
    num_frames = clip.shape[2]


    # convert to 4D tensor (batch number is always set to 1 for each batch of clips)
    clip_4d = clip.permute(0, 2, 1, 3, 4).reshape(-1, clip.shape[2], clip.shape[3], clip.shape[4])


    # unpack each batch 
    for i in range(num_frames): 

        #cycles through each frame
        frame = clip_4d[:, i, :, :].unsqueeze(0) 
     
        #generate prediction
        with torch.no_grad():
            logits = model(frame)  
            probability = torch.sigmoid(logits)  
            predictions.append(probability.item())


    #average prediction score (0 to 1) 
    avg_pred = mean(predictions) 


    #return decision for batch 
    if avg_pred < 0.5: 
        return 0
    
    else: 
        return 1