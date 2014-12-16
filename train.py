from DBN import DBN
from dbn_util import * 
import pickle
import sys

train_file = sys.argv[1]
model_file = sys.argv[2]

d,l=loadDataSet(train_file,784)
d=d[0:100]
l=l[0:100]
#testd=testd[0:10]
#testl=testl[0:10]
model = DBN([20,10]) 
model.trainDBN(d,l,batchsize=[20,10],numepoch=[10,20])
file = open(model_file,"wb")
pickle.dump(model,file)
model.Print()
file.close()
