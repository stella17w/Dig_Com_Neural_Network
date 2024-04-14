import pickle
import numpy
import random
import io

with open("2016.04C.multisnr.pkl", 'rb') as file:
    data = pickle.load(file,encoding='bytes')

test={}
validation={}

#pick 20% and split into validation and test
num=int(0.1*len(data))
for i in range(0,num):
    rand_num=random.randint(0,len(data)-1)
    keys=[key for key in data]
    rand_key=keys[rand_num]
    test[rand_key]=data[rand_key]
    data.pop(rand_key)
    rand_num=random.randint(0,len(data)-1)
    keys=[key for key in data]
    rand_key=keys[rand_num]
    validation[rand_key]=data[rand_key]
    data.pop(rand_key)

#pickle the sets into training, test, and validation
with open("training.pkl",'wb') as file:
    pickle.dump(data,file)
with open("test.pkl","wb") as file:
    pickle.dump(test,file)
with open("validation.pkl","wb") as file:
    pickle.dump(validation,file)
