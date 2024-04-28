import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random

#dictionary to convert strings into numberic values
radio_mod={'BPSK': 0,
           'QPSK': 1,
           '8PSK': 2,
           'QAM16':3,
           'QAM64':4,
           'PAM4':5,
           'BFSK':6,
           'CPFSK':7,
           'GFSK':8,
           'WBFM':9,
           'AM-SSB':10,
           'AM-DSB':11}

#~Define the model
class RadioNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1=nn.BatchNorm1d(2)
        self.conv1=nn.Conv1d(2,64,9)
        self.act1=nn.ReLU()
        self.conv2=nn.Conv1d(64,64,9)
        self.act2=nn.ReLU()
        self.conv3=nn.Conv1d(64,12,9)
        self.act3=nn.ReLU()
        self.pool=nn.MaxPool1d(9)
        self.hidden2=nn.BatchNorm1d(12)
        self.linear1=nn.Linear(11,64)
        self.hidden3=nn.BatchNorm1d(12)
        self.linear2=nn.Linear(64,256)
        self.hidden4=nn.BatchNorm1d(12)
        self.linear3=nn.Linear(256,1)
        self.hidden5=nn.BatchNorm1d(12)
        self.output=nn.Softmax(1)

    def forward(self, x):
        x = self.act1(self.conv1(self.hidden1(x)))
        x = self.act2(self.conv2(x))
        x = self.pool(self.act3(self.conv3(x)))
        x = self.hidden5(self.linear3(self.hidden4(self.linear2(self.hidden3(self.linear1(self.hidden2(x)))))))
        x = self.output(x)
        return x[:,:,-1]

model=RadioNN()

#~Training

model.conv1.reset_parameters()
model.conv2.reset_parameters()
model.conv3.reset_parameters()
model.linear1.reset_parameters()
model.linear2.reset_parameters()
model.linear3.reset_parameters()

#loss and optimizer
fn_loss=nn.CrossEntropyLoss() #cross entorpy loss
optimizer = optim.Adam(model.parameters(), lr=0.0009)

#load training data
with open("training.pkl", 'rb') as file:
    train_data = pickle.load(file,encoding='bytes')
#split into x and y
keys=[key for key in train_data]
x_train=[]
y_train=[]
for k in keys:
    for x in train_data[k]:
        if len(x_train)<2:
            rand_num=0
        else:
            rand_num=random.randint(0,len(x_train)-1)
        x_train.insert(rand_num,x)
        y_train.insert(rand_num,radio_mod[k[0].decode()])
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
        if len(x_valid)<2:
            rand_num=0
        else:
            rand_num=random.randint(0,len(x_valid)-1)
        x_valid.insert(rand_num,x)
        y_valid.insert(rand_num,radio_mod[k[0].decode()])
x_valid=torch.Tensor(np.array(x_valid))
y_valid=torch.Tensor(np.array(y_valid))

#training loop

mini_batch_size=64
first_accurary=-3
previous_accurary=-2
current_accurary=-1
y_pred = model(x_valid[0:len(x_valid)])
current_loss=fn_loss(y_pred,y_valid.long())
v_loss=[current_loss.detach().float()]
v_accurary=[(torch.argmax(y_pred) == y_valid).float().mean()*100]
y_pred = model(x_train[0:len(x_train)])
current_loss=fn_loss(y_pred,y_train.long())
t_loss=[current_loss.detach().float()]
t_accurary=[(torch.argmax(y_pred,dim=1) == y_train).float().mean()*100]

#add stopping critera
while current_accurary>previous_accurary or current_accurary>first_accurary:
    for i in range(0, len(train_data), mini_batch_size):
        xbatch = x_train[i:i+mini_batch_size]
        ypred=model(xbatch)
        ybatch = y_train[i:i+mini_batch_size]
        current_loss=fn_loss(ypred, ybatch.long())
        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()
    #calcluate loss and accurary
    first_accurary=previous_accurary
    previous_accurary=current_accurary
    y_pred = model(x_valid[0:len(x_valid)])
    current_loss=fn_loss(y_pred,y_valid.long())
    current_accurary=(torch.argmax(y_pred,dim=1) == y_valid).float().mean()*100
    print(current_accurary)
    v_accurary.append(current_accurary)
    v_loss.append(current_loss.detach().float())
    y_pred = model(x_train[0:len(x_train)])
    current_loss=fn_loss(y_pred,y_train.long())
    t_loss.append(current_loss.detach().float())
    t_accurary.append((torch.argmax(y_pred,dim=1) == y_train).float().mean()*100)
    

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
        if len(x_test)<2:
            rand_num=0
        else:
            rand_num=random.randint(0,len(x_test)-1)
        x_test.insert(rand_num,x)
        y_test.insert(rand_num,radio_mod[k[0].decode()])
x_test=torch.Tensor(np.array(x_test))
y_test=torch.Tensor(np.array(y_test))

y_pred = model(x_test[0:len(x_test)])
final_accurary=(torch.argmax(y_pred,dim=1) == y_test).float().mean()*100
print("Final accuary:",final_accurary.float())

# 3 predictions

#analog vs digital
rand_int=random.randint(0,len(x_test))
current_pred=y_pred[rand_int]
digital=sum(current_pred[0:10])
analog=sum(current_pred[10:])
print("Analog:", analog, "Digtial", digital)
print(list(radio_mod.keys())[y_test[rand_int].int()])
plt.scatter(x_test[rand_int][0],x_test[rand_int][1])
plt.show()

#Phase shift keying
rand_int=random.randint(0,len(x_test))
current_pred=y_pred[rand_int]
psk=sum(current_pred[0:3])
other=sum(current_pred[3:])
print("PSK:", psk, "Other", other)
print(list(radio_mod.keys())[y_test[rand_int].int()])
plt.scatter(x_test[rand_int][0],x_test[rand_int][1])
plt.show()

#Frequency shift keying
rand_int=random.randint(0,len(x_test))
current_pred=y_pred[rand_int]
fsk=sum(current_pred[6:9])
other=1-fsk
print("FSK:", fsk, "Other", other)
print(list(radio_mod.keys())[y_test[rand_int].int()])
plt.scatter(x_test[rand_int][0],x_test[rand_int][1])
plt.show()


#delete model
del model