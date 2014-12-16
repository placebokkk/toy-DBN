from DBN import DBN
from dbn_util import * 

d,l=loadDataSet("train.txt",784)
#d=d[0:100]
#l=l[0:100]
testd,testl=loadDataSet("test.txt",784)
#testd=testd[0:10]
#testl=testl[0:10]
testD = DBN([100,100]) 
testD.trainDBN(d,l,batchsize=[200,20],numepoch=[50,200])
testD.Print()
r=[]
for i in range(0,len(testd)): 
    r.append(testD.predict(testd[i]))
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