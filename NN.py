import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

#dictionary to convert strings into numberic values
radio_mod={'BPSK':0,'QPSK':1,'8PSK':2,'QAM16':3,'QAM64':4,'BFSK':5,'CPFSK':6,'PAM4':7,'WBFM':8,'AM-SSB':9,'AM-DSB':10,'GFSK':11}

#~Define the model
model=nn.Sequential(
    nn.Conv1d(2,31,11),
    nn.ReLU(),
    nn.Conv1d(31,31,11),
    nn.ReLU(),
    nn.Conv1d(31,64,11),
    nn.ReLU(),
    nn.Linear(98,64),
    nn.Linear(64,227),
    nn.Linear(227,1),
    nn.MaxPool1d(1),
    nn.Softmax(1)
)

#~Training

#loss and optimizer
fn_loss = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

#load training data
with open("training.pkl", 'rb') as file:
    train_data = pickle.load(file,encoding='bytes')
#split into x and y
keys=[key for key in train_data]
x_train=[]
y_train=[]
for k in keys:
    for x in train_data[k]:
        x_train.append(x)
        y_train.append([radio_mod[k[0].decode()],k[1]])
x_train=torch.Tensor(x_train)
y_train=torch.Tensor(y_train)

#load validation data
with open("validation.pkl",'rb') as file:
    valid_data=pickle.load(file,encoding='bytes')
#split into x and y
keys=[key for key in valid_data]
x_valid=[]
y_valid=[]
for k in keys:
    for x in valid_data[k]:
        x_valid.append(x)
        y_valid.append([radio_mod[k[0].decode()],k[1]])
x_valid=torch.Tensor(x_valid)
y_valid=torch.Tensor(y_valid)

#training loop
mini_batch_size=64
previous_accurary=-1
current_accurary=0
accurary=[]
loss=[]

#add stopping critera
while current_accurary>previous_accurary:
    for i in range(0, len(train_data), mini_batch_size):
        xbatch = x_train[i:i+mini_batch_size]
        ypred=model(xbatch)
        ybatch = y_train[i:i+mini_batch_size]
        current_loss = fn_loss(ypred, ybatch)
        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()
    #calcluate loss and accurary
    loss.append(current_loss)
    previous_accurary=current_accurary
    with torch.no_grad():
        y_pred = model(x_valid[0:len(x_valid)])
    current_accuracy = (ypred.round() == y_valid).float().mean()
    accurary.append(current_accurary)
print(loss)
print(accurary)