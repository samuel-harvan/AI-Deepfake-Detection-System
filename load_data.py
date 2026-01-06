from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import cv2
from PIL import Image
import json
from getdata import find_frames


# defining custom dataset loader for transformations/file loading
class dataset_loader(Dataset): 

    def __init__(self, frame_paths, labels, transform=None): 

        self.frame_paths = frame_paths 
        self.labels = labels 
        self.transform = transform 


    def __len__(self): 

        return len(self.frame_paths) 
    

    def __getitem__(self, index): 

        image_arr = cv2.imread(self.frame_paths[index]) 
        image = Image.fromarray(image_arr)
        image = self.transform(image) 
        label = self.labels[index]


        return image, label 





def create_tdata() -> tuple[DataLoader, DataLoader, DataLoader]: 


    # defining transformations 

    train_trans = transforms.Compose([
        transforms.Resize((299, 299)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), 
        transforms.RandomHorizontalFlip() # to prevent overfitting
    ])

    test_trans = transforms.Compose([
        transforms.Resize((299, 299)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


    # load stored data 
    with open("stored_data.json", "r") as var_file: 

        data = json.load(var_file) 


    # CHANGE THIS: WE NEED 1000/1000 SPLIT FOR fake:real. RIGHT NOW ITS 2000/1000, fake:real
    # X: 20000 frames, 50/50 split (fake vs. real) 
    X = data["paths"]
    y = data["labels"]


    # splitting data into training, testing, and validation datasets 
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=41) 
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=41) 


    # creating datasets 
    train_dataset = dataset_loader(X_train, y_train, transform=train_trans)
    test_dataset = dataset_loader(X_test, y_test, transform=test_trans)
    val_dataset = dataset_loader(X_val, y_val, transform=test_trans)

    
    # creating dataloaders 
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True) 
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) 
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False) 


    return train_loader, test_loader, val_loader