from DBN import DBN
from dbn_util import * 
import pickle

test_str = sys.argv[1]
model_str = sys.argv[2]
testd,testl=loadDataSet(test_str,784)
file = open(model_str,"rb")
model = pickle.load(file) 

model.Print()
r=[]
for i in range(0,len(testd)): 
    r.append(model.predict(testd[i]))
print testl
c=map(lambda x:x+1,map(argmax,r))
print c
dif = array(c)-array(testl)
print dif
rate=0.0
for i in range(0,len(testd)):
    if dif[i]==0:
        rate+=1
print rate/len(testd)

file.close()