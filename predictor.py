import torch 
from torchvision import transforms
from network.xception_model import xception
from detection import read_vid
from typing import Union
from PIL import Image


# input type: file path
def predict(file_path) -> Union[float, None]: 

    tensor_lst = [] 


    # find available device on current running computer

    if torch.cuda.is_available(): 

        device = torch.device("cuda")


    elif torch.backends.mps.is_available(): 

        device = torch.device("mps") 

    
    else: 

        device = torch.device("cpu")


    # load model and weights 
    model = xception(pretrained=False, num_classes=2) 
    model.load_state_dict(torch.load("network/trained_weights.pth", map_location=device))
    # will upload weights to github and use that repo instead (public access) 


    model.eval() 


    eval_trans = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(std=(0.5, 0.5, 0.5), mean=(0.5, 0.5, 0.5))
    ])


    # extract face frames
    frames = read_vid(file_path, 50) 


    # if faces cannot be found, end the program
    if len(frames) == 0: 
        return None


    for frame in frames: 

        # apply necessary transformations
        frame_pil = Image.fromarray(frame) 
        frame_tensor = eval_trans(frame_pil) 
        tensor_lst.append(frame_tensor) 


    # create batch
    tensor_lst = torch.stack(tensor_lst)


    with torch.no_grad(): 
        
        # get probability
        pred = model(tensor_lst) 
        prob = torch.softmax(pred, dim=1) 


        # output probability for fake for frame batch
        prob_fake = prob[:, 1]
        total_prob = torch.mean(prob_fake) 


    return round(total_prob.item()*100, 4) 