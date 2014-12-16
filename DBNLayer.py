from numpy import *
from dbn_util import * 
class DBNLayer:
    def __init__(self,name="Layer"):
        self.name=name
        
    def Out(self):
        return self.out
        
    def train(self,data,numhid,batchsize,numepoch,eta=0.1,penalty=2e-4,momentum=0.5):
        print "start training" + self.name 
        X = mat(data)
       
        numcases,numdims = shape(X) 
        WX = array(random.normal(size=numdims*numhid)).reshape(numdims,numhid)*0.1

        bX = zeros([1,numdims])
        bH = mat(zeros([1,numhid]))
        '''   
        probH = zeros([numcases,numhid]);
        probX_rec = zeros([numcases,numdims]);
        probH_rec = zeros([numcases,numhid]);
        
        sampleH = zeros([numcases,numhid]);
        sampleX_rec = zeros([numcases,numdims]);
        sampleH_rec = zeros([numcases,numhid]);
        '''
        WXinc  = zeros([numdims,numhid]);
        
        bXinc = zeros([1,numdims]);    
        bHinc = zeros([1,numhid]);
              
        maxbatch = numcases/batchsize    
        for epoch in range(0,numepoch):
            print epoch
            for batch in range(0,maxbatch+1):
                spos =  batch  * batchsize
                epos =  (batch +1) * batchsize
                if batch  == maxbatch:
                    epos = numcases

                batchnum = epos -spos
                if batchnum <=0:
                    continue
                
                data = X[spos:epos]
                
                #go up,calculate the Hidden node prob conditioned on the X
                probH = sigmoid(data*WX  + repeat(bH.transpose(),batchnum,1).transpose())
                rA = array(random.uniform(0,1,batchnum*numhid)).reshape(batchnum,numhid)
                sampleH = mat(map(lambda x:map(int,x),array(probH >rA)))

                #go down
                #feature layer
                 
                probX_rec = sigmoid(sampleH*WX.transpose() + repeat(bX.transpose(),batchnum,1).transpose())
                rA = array(random.uniform(0,1,batchnum*numdims)).reshape(batchnum,numdims)
                sampleX_rec = mat(map(lambda x:map(int,x),array(probX_rec >rA)))

                #go up again to Hidden Layer
                probH_rec = sigmoid(sampleX_rec*WX  + repeat(bH.transpose(),batchnum,1).transpose())
                rA = array(random.uniform(0,1,batchnum*numhid)).reshape(batchnum,numhid)
                sampleH_rec = mat(map(lambda x:map(int,x),array(probH_rec >rA)))
                
                
                #update the parameters   
                #deltaW = v0h0-v1h1 or deltaW = v0p0-v1p1,
                #where h means the sample value and p means the prob value
                #the later could speed the learning process.
                dWX = data.transpose()*probH - sampleX_rec.transpose()*probH_rec
                dbX = data.sum(axis = 0) - sampleX_rec.sum(axis = 0)
                dbH = probH.sum(axis = 0) - probH_rec.sum(axis = 0)
                '''
                print "WX2";
                print WX
                '''
                #@penalty used for regularization
                #@momentum see hinton's -A Practical Guide to Training RBM
                #@eta is the speed of learning
                #
                WXinc = momentum*WXinc + eta*(dWX/batchnum - penalty*WX);
                bXinc = momentum*bXinc + eta*(dbX/batchnum);
                bHinc = momentum*bHinc + eta*(dbH/batchnum);
                '''
                print "WXinc";
                print WXinc
                '''
                WX = WX + WXinc;
                bX = bX + bXinc;
                bH = bH + bHinc;
     
        self.WX = WX
        self.bX = bX
        self.bH = bH
        self.out = sigmoid(X*WX  + repeat(bH.transpose(),numcases,1).transpose())
    
    def predict(self,x):

        out =  sigmoid(x*self.WX+self.bH)
        return out
