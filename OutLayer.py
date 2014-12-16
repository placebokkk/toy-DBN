from numpy import *
from dbn_util import * 
class OutLayer:
    def __init__(self,name="OutLayer"):
        self.name=name
        
    def predict(self,x):
        prob =  sigmoid(mat(x)*self.WX+self.bH)
        out = softmax(prob*self.WL.transpose()+self.bL)
        return out
        
    def train(self,data,labels,numhid,batchsize,numepoch,eta=0.1,penalty=2e-4,momentum=0.5):
        X = mat(data)
        L = mat(convOneofK(labels))
        #print L

        numcases,numdims = shape(X) 
        numlabels = len(set(labels))
        WX = array(random.normal(size=numdims*numhid)).reshape(numdims,numhid)*0.1
        WL =  array(random.normal(size =numlabels*numhid )).reshape(numlabels,numhid)*0.1

        bX = zeros([1,numdims])
        bH = mat(zeros([1,numhid]))
        bL = mat(zeros([1,numlabels]))
        '''   
        probH = zeros([numcases,numhid]);
        probX_rec = zeros([numcases,numdims]);
        probL_rec = zeros([numcases,numlabels]);
        probH_rec = zeros([numcases,numhid]);
        sampleH = zeros([numcases,numhid]);
        sampleX_rec = zeros([numcases,numdims]);
        sampleL_rec = zeros([numcases,numlabels]);
        sampleH_rec = zeros([numcases,numhid]);
        '''
        WXinc  = zeros([numdims,numhid]);
        WLinc = zeros([numlabels,numhid]);

        bXinc = zeros([1,numdims]);    
        bHinc = zeros([1,numhid]);
        bLinc = zeros([1,numlabels]);


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
                labels = L[spos:epos]
                
                #go up,calculate the Hidden node prob conditioned on the X and L layers
                probH = sigmoid(data*WX + labels*WL + repeat(bH.transpose(),batchnum,1).transpose())
                rA = array(random.uniform(0,1,batchnum*numhid)).reshape(batchnum,numhid)
                sampleH = mat(map(lambda x:map(int,x),array(probH >rA)))



                #go down
                #feature layer
                
                
                probX_rec = sigmoid(sampleH*WX.transpose() + repeat(bX.transpose(),batchnum,1).transpose())
                rA = array(random.uniform(0,1,batchnum*numdims)).reshape(batchnum,numdims)
                sampleX_rec = mat(map(lambda x:map(int,x),array(probX_rec >rA)))
                #label layer,softmax here.
                #probL_rec = sampleH*WL.transpose() + repeat(bL.transpose(),batchnum,1).transpose()
                #sampleL_rec = mat(softmaxSample(array(probL_rec)))

                probL_rec = exp(sampleH*WL.transpose() + repeat(bL.transpose(),batchnum,1).transpose())
                sampleL_rec = mat(softmaxSample2(array(probL_rec)))
                
                #go up again to Hidden Layer
                probH_rec = sigmoid(sampleX_rec*WX + sampleL_rec*WL + repeat(bH.transpose(),batchnum,1).transpose())
                rA = array(random.uniform(0,1,batchnum*numhid)).reshape(batchnum,numhid)
                sampleH_rec = mat(map(lambda x:map(int,x),array(probH_rec >rA)))
                
                
                #update the parameters   
                #deltaW = v0h0-v1h1 or deltaW = v0p0-v1p1,
                #where h means the sample value and p means the prob value
                #the later could speed the learning process.
                dWX = data.transpose()*probH - sampleX_rec.transpose()*probH_rec
                dWL = labels.transpose()*probH - sampleL_rec.transpose()*probH_rec
                dbX = data.sum(axis = 0) - sampleX_rec.sum(axis = 0)
                dbH = probH.sum(axis = 0) - probH_rec.sum(axis = 0)
                dbL = labels.sum(axis = 0) - sampleL_rec.sum(axis = 0)
                '''
                print "WX2";
                print WX
                '''
                #@penalty used for regularization
                #@momentum see hinton's -A Practical Guide to Training RBM
                #@eta is the speed of learning
                #
                WXinc = momentum*WXinc + eta*(dWX/batchnum - penalty*WX);
                WLinc = momentum*WLinc + eta*(dWL/batchnum - penalty*WL);
                bXinc = momentum*bXinc + eta*(dbX/batchnum);
                bHinc = momentum*bHinc + eta*(dbH/batchnum);
                bLinc = momentum*bLinc + eta*(dbL/batchnum);
                '''
                print "WXinc";
                print WXinc
                '''
                WX = WX + WXinc;
                bX = bX + bXinc;
                bH = bH + bHinc;
                WL = WL + WLinc;
                bL = bL + bLinc;




        self.WX = WX
        self.WL = WL
        self.bX = bX
        self.bH = bH
        self.bL = bL