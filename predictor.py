# Need to fix up some issues with predictions


import torch 
from network.xception_model import xception
import numpy as np 


def predict(clips, batch_num) -> int: 

    # load model and weights 
    model = xception(num_classes=2, pretrained=None) 


    # need to detect if mps is available else you have to switch to cuda or cpu (cross platform support)
    #load_weight = torch.load("network/trained_weights.pth", map_location=torch.device("mps"))
    #model.load_state_dict(load_weight)


    model.eval() 


    predictions = []


    # load 5D tensor and calculate frames per batch
    clip = clips[batch_num]
    num_frames = clip.shape[2]


    # convert to 4D tensor (batch number is always set to 1 for each batch of clips)
    clip_4d = clip.permute(0, 2, 1, 3, 4).reshape(-1, clip.shape[2], clip.shape[3], clip.shape[4])


    # unpack each batch 
    for i in range(num_frames): 

        # cycles through each frame
        frame = clip_4d[:, i, :, :].unsqueeze(0) 
     

        # generate prediction
        with torch.no_grad():
            
            logits = model(frame)  
            probability = torch.sigmoid(logits)  
            predictions.append(probability.item())


    # average prediction score (0 to 1) 
    avg_pred = np.mean(predictions) 


    # return decision for batch 
    if avg_pred > 0.5: 
        return 1
    
    else: 
        return 0