from DBNLayer import DBNLayer
from OutLayer import OutLayer
from numpy import *
class DBN:
    def __init__(self,hidlist,config =0):

        self.hidlist = hidlist
        self.hlnum = len(hidlist)

        
    def trainDBN(self,data,labels,batchsize,numepoch,config =0):
        hLayer = [0,0,0,0,0,0] 
        for i in range(0,self.hlnum-1):
            hLayer[i] = DBNLayer("layer"+str(i))
            if i==0:
                input = data
            else:
                input = hLayer[i-1].Out()
            print "start training" + str(i) +"layer"
            hLayer[i].train(input,self.hidlist[i],batchsize[i],numepoch[i])
            print "finish training" + str(i) +"layer"
        oLayer = OutLayer()
        print "start training out layer"
        oLayer.train(hLayer[self.hlnum-2].Out(),labels,self.hidlist[self.hlnum-1],batchsize[-1],numepoch[-1])
        print "finish training out layer"
        self.hLayer = hLayer
        self.oLayer = oLayer
    
    def Print(self):
        for i in range(0,self.hlnum-1):
            print self.hLayer[i].WX
             
        print self.oLayer.WX
        print self.oLayer.WL
        
    def predict(self,x):
        for i in range(0,self.hlnum-1):            
            if i==0:
                prob=self.hLayer[i].predict(x)
                #print "prob1"
                #print prob
            else:
                prob=self.hLayer[i].predict(prob)
                #print "prob2"
                #print prob
           
        out=self.oLayer.predict(prob)
        return out;

    