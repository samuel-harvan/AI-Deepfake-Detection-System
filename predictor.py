import torch 
import cv2
from torchvision import transforms
from network.xception_model import xception
from detection import read_vid


# input type: video file
def predict(file_path) -> float | None: 


    if torch.cuda.is_available(): 

        device = torch.device("cuda")


    elif torch.backends.mps.is_available(): 

        device = torch.device("mps") 

    
    else: 

        device = torch.device("cpu")


    # load model and weights 
    model = xception(pretrained=False, num_classes=2) 
    model.load_state_dict(torch.load("network/trained_weights.pth"), map_location=device)
    # will upload weights to github and use that repo instead


    # need to detect if mps is available else you have to switch to cuda or cpu (cross platform support)
    #load_weight = torch.load("network/trained_weights.pth", map_location=torch.device("mps"))
    #model.load_state_dict(load_weight)


    model.eval() 


    eval_trans = transforms.Compose(
        transforms.Resize(299, 299),
        transforms.ToTensor(),
        transforms.Normalize(std=(0.5, 0.5, 0.5), mean=(0.5, 0.5, 0.5))
    )


    frames = read_vid(file_path, 50) 


    # if faces cannot be found, end the program
    if len(frames) == 0: 
        return None


    for frame in frames: 

        frame = eval_trans(frame) 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame) 


    frames = torch.stack(frames).to(device) 


    with torch.no_grad: 
        
        # get probability
        pred = model(frames) 
        prob = torch.softmax(pred, dim=1) 


        prob_fake = prob[:, 1]


        total_prob = torch.mean(prob_fake) 


    return round(total_prob*100, 4) 