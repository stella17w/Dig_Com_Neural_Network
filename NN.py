import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

#dictionary to convert strings into numberic values
radio_mod={'BPSK':0,'QPSK':1,'8PSK':2,'QAM16':3,'QAM64':4,'BFSK':5,'CPFSK':6,'PAM4':7,'GFSK':8,'WBFM':9,'AM-SSB':10,'AM-DSB':11}

#~Define the model
class RadioNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1=nn.InstanceNorm1d(2)
        self.conv1=nn.Conv1d(2,64,11)
        self.act1=nn.ReLU()
        self.hidden2=nn.InstanceNorm1d(64)
        self.conv2=nn.Conv1d(64,2,11)
        self.act2=nn.ReLU()
        self.hidden3=nn.InstanceNorm1d(2)
        self.linear1=nn.Linear(108,64)
        self.linear2=nn.Linear(64,256)
        self.linear3=nn.Linear(256,2)
        self.pool=nn.MaxPool1d(1)
        self.output=nn.Softmax(1)

    def forward(self, x):
        x = self.act1(self.conv1(self.hidden1(x)))
        x = self.act2(self.conv2(self.hidden2(x)))
        x = self.linear3(self.linear2(self.linear1(self.hidden3(x))))
        x = self.output(self.pool(x))
        return x [:, -1, :]

model=RadioNN()

#~Training

#loss and optimizer
fn_loss=nn.CrossEntropyLoss() #cross entropy
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
x_train=torch.Tensor(np.array(x_train))
y_train=torch.Tensor(np.array(y_train))

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
x_valid=torch.Tensor(np.array(x_valid))
y_valid=torch.Tensor(np.array(y_valid))

#training loop

mini_batch_size=64
previous_accurary=-2
current_accurary=-1
y_pred = model(x_valid[0:len(x_valid)])
v_accurary=[(y_pred.round() == y_valid).float().mean()*100]
current_loss=fn_loss(y_pred,y_valid)
v_loss=[current_loss.detach().float()]
y_pred = model(x_train[0:len(x_train)])
t_accurary=[(y_pred.round() == y_train).float().mean()*100]
current_loss=fn_loss(y_pred,y_train)
t_loss=[current_loss.detach().float()]

#add stopping critera
while current_accurary>previous_accurary:
    for i in range(0, len(train_data), mini_batch_size):
        xbatch = x_train[i:i+mini_batch_size]
        ypred=model(xbatch)
        ybatch = y_train[i:i+mini_batch_size]
        current_loss=fn_loss(ypred, ybatch)
        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()
    #calcluate loss and accurary
    previous_accurary=current_accurary
    y_pred = model(x_valid[0:len(x_valid)])
    current_accurary=(y_pred.round() == y_valid).float().mean()*100
    v_accurary.append(current_accurary)
    current_loss=fn_loss(y_pred,y_valid)
    v_loss.append(current_loss.detach().float())
    y_pred = model(x_train[0:len(x_train)])
    t_accurary.append((y_pred.round() == y_train).float().mean()*100)
    current_loss=fn_loss(y_pred,y_train)
    t_loss.append(current_loss.detach().float())

#plot
plt.plot(range(0,len(t_accurary)),t_accurary,range(0,len(v_accurary)),v_accurary)
plt.xlabel('epochs')
plt.ylabel('accurary')
plt.legend(["training","validation"])
plt.show()

plt.plot(range(0,len(t_loss)),t_loss,range(0,len(v_loss)),v_loss)
plt.xlabel('epochs')
plt.ylabel('Cross Entropy loss')
plt.legend(["training","validation"])
plt.show()

#~performance of test set

#loading test set
with open("test.pkl", 'rb') as file:
    test_data = pickle.load(file,encoding='bytes')
#split into x and y
keys=[key for key in test_data]
x_test=[]
y_test=[]
for k in keys:
    for x in test_data[k]:
        x_test.append(x)
        y_test.append([radio_mod[k[0].decode()],k[1]])
x_test=torch.Tensor(np.array(x_test))
y_test=torch.Tensor(np.array(y_test))

y_pred = model(x_test[0:len(x_test)])
final_accurary=(y_pred.round() == y_test).float().mean()*100
print("Final accuuary:",final_accurary.float())

#delete model
del model