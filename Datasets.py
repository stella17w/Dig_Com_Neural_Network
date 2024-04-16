import pickle
import numpy
import random
import io

with open("2016.04C.multisnr.pkl", 'rb') as file:
    data = pickle.load(file,encoding='bytes')

test={}
validation={}
keys=[key for key in data]

#pick 20% and split into validation and test
for k in keys:
    num=int(0.1*len(data))
    for i in range(0,num):
        rand_num=random.randint(0,len(data[k])-1)
        if i==0:
            test[k]=[data[k][rand_num]]
        else:
            test[k].append(data[k][rand_num])
        numpy.delete(data[k],rand_num)
        rand_num=random.randint(0,len(data[k])-1)
        if i==0:
            validation[k]=[data[k][rand_num]]
        else:
            validation[k].append(data[k][rand_num])
        numpy.delete(data[k],rand_num)

#pickle the sets into training, test, and validation
with open("training.pkl",'wb') as file:
    pickle.dump(data,file)
with open("test.pkl","wb") as file:
    pickle.dump(test,file)
with open("validation.pkl","wb") as file:
    pickle.dump(validation,file)
