from numpy import *

'''
Fit an RBM to discrete labels 
code by YangChao
a python version based on matlab version of Andrej Karpathy
'''
def loadDataSet(fs,dim):
    data = []
    label = []
    fr = open(fs)
    for line in fr.readlines():
        lineArr = line.strip().split(" ")
        #if len(lineArr)!= dim:
        #    pass;
        data.append(map(float,lineArr[0:dim]))
        label.append(int(lineArr[dim]))
    return data,label

def sigmoid(X):
    return 1.0/(1+exp(-X))
    
#use 1-of-K representation    
def convOneofK(X):
    numlabels=len(set(X))
    numcases =len(X)
    Y = zeros([numcases,numlabels])
    for i in range(0,numcases):
        Y[i][X[i]-1]=1
        
    return Y
    

#an simple sample,choose the max to be 1,others 0
def softmaxSample(M):
    m,n = M.shape
    R = zeros([m,n])
    for i in range(0,m):
        R[i][M[i].argmax()]=1
    return R
    
def softmax(X):
    M = array(exp(X))
    m,n = M.shape
    R = zeros([m,n])
    N = M/sum(M,1).reshape(m,1)    
    return N
    
#sample method from Andrej Karpathy's code
def softmaxSample2(M):
    m,n = M.shape
    R = zeros([m,n])
    N = M/sum(M,1).reshape(m,1)
    for i in range(0,m):
        sample=cumsum(N[i])
        r= random.uniform()
        sample=array((map(int,sample > r)))             
        R[i][sample.argmax()]=1
    return R


