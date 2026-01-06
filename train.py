import torch
import numpy as np
import torch.nn as nn
from xception_model import xception
from load_data import create_tdata
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt 


# data loaders 
train_loader, test_loader, val_loader = create_tdata() 


# Load Xception model with pretrained ImageNet weights
model = xception(pretrained=True, num_classes=1000)     


# Only training classifier layer

# Freeze all layers 
for param in model.parameters(): 
    param.requires_grad = False


# Unfreeze classifier
for param in model.fc.parameters(): 
    param.requires_grad = True


# Set number of classes to 2
model.fc = nn.Linear(2048, 2) 


# Create optimizer/criterion  
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss() 


# Move model to GPU 
model = model.to(torch.device("mps")) 


model.train()
losses = [] 
avg_loss = []





def train(model, train_loader, optimizer, criterion, num_epochs):

    best_val = float("infinity") 


    # training loop 
    for epoch in range(num_epochs): 

        for frames, labels in train_loader: 

            # labels are converted to long tensors (for CE loss) 
            frames, labels = frames.to(torch.device("mps")), labels.to(torch.device("mps")).long() 


            # get prediction
            y_pred = model(frames)  


            # measure loss
            loss = criterion(y_pred, labels) 
            losses.append(loss.item()) 


            # backpropagation 
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step()


        avg_loss.append(np.mean(losses))


        # validation 

        model.eval()


        with torch.no_grad(): 

            for frames, labels in val_loader: 

                frames, labels = frames.to(torch.device("mps")), labels.to(torch.device("mps")).long()


                # get predictions   
                y_val = model(frames) 

                
                # Find error
                val_loss = criterion(y_val, labels) 


                if val_loss < best_val: 

                    best_val = val_loss


                    # save weights
                    torch.save(model.state_dict(), "trained_weights.pth")


                else: 

                    continue
        

        print(f"Final loss during training: {losses[-1]:.5f}")
        
        
    plt.plot(range(num_epochs), avg_loss)
    plt.ylabel("Avg. Loss/Error")
    plt.xlabel("Epoch") 
    plt.show() 


# testing
def test(model, test_loader): 

    model.eval()


    all_probs = [] 
    labels = [] 


    correct = 0
    total = 0


    with torch.no_grad(): 

        for data, target in test_loader: 

            data, target = data.to(torch.device("mps")), target.to(torch.device("mps")).long() 


            # get prediction 
            y_eval = model(data) 


            # probabilities 
            prob = torch.softmax(y_eval, dim=1)
            pred_class = torch.argmax(prob, dim=1)


            # find probabilities for positive class for each batch
            fake_probs = prob[:, 1].cpu().numpy()  


            all_probs.extend(fake_probs) 
            labels.extend(target.cpu().numpy())


            # grade model every batch 
            for i, pred in enumerate(pred_class): 
                
                if pred == target[i]: 

                    correct +=1
                    total +=1
                

                else: 

                    total+=1 
    

    accuracy = correct/total*100


    roc_auc = roc_auc_score(labels, all_probs)


    print(f"Model accuracy during testing was: {accuracy}")
    print(f"roc-auc: {roc_auc}")





if __name__ == "__main__": 

    # ten epochs of training
    train(model, train_loader, optimizer, criterion, 10)


    # pause before testing
    print("Press 'ENTER' to continue to testing")
    input()


    # testing 
    test(model, test_loader, optimizer, criterion) 