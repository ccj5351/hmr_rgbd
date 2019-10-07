from model import xyz2smplRegressor,xyz2lspSmplRegressor
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
import os,dataloader,util,json
import numpy as np
from collections import OrderedDict
from vis import Visualizations
import config as cfg
from SMPL_layer import SMPL_Layer

def createDataLoaderAdv():
    mosh=dataloader.MoSH()
    dataSize=len(mosh)
    dataInd=list(range(dataSize))
    dataSplit=int(np.floor(cfg.validationSplitFrac*dataSize))
    np.random.seed(cfg.randomSeed)
    np.random.shuffle(dataInd)
    trainInd,valInd=dataInd[dataSplit:],dataInd[:dataSplit]
    trainSampler=SubsetRandomSampler(trainInd)
    validSampler=SubsetRandomSampler(valInd)
    
    trainDataLoader=DataLoader(dataset=mosh,batch_size=1,sampler=trainSampler,drop_last=True)
    validDataLoader=DataLoader(dataset=mosh,batch_size=1,sampler=validSampler,drop_last=True)
    
    return trainDataLoader,validDataLoader

smplLayer=SMPL_Layer(cfg.smplGender,cfg.smplModelPath)
smplLayer=smplLayer.cuda()
plotVis=Visualizations()
plotVis.savePath=cfg.inferModelPath
plotVis.numVis=1


saveString="human36mTestImgVis"
#saveString="valImgVis"

human36m=dataloader.Human36M()
dl=DataLoader(dataset=human36m,batch_size=1,shuffle=True)
itr=iter(dl)

#dataLoaderAdvTrain,dataLoaderAdvVal=createDataLoaderAdv()
#itr=iter(dataLoaderAdvVal)

def loadModel(model,stateDict):
    newStateDict=OrderedDict()
    for k,v in stateDict.items():
        if "regressor.meanParam" in k:
            continue
        newStateDict[k]=v
    model.load_state_dict(newStateDict)
    return model

def main():
    #inferModel=xyz2smplRegressor().cuda() 
    inferModel=xyz2lspSmplRegressor().cuda() 
    encoderModelPath=os.path.join(cfg.inferModelPath,"smplRegressor145000.pt")
    loadModel(inferModel,torch.load(encoderModelPath))
    inferModel.eval()
    for r in range(1000):
        currData=next(itr)
        currTheta=currData["theta"].cuda()
        currTheta=currTheta[0]
        currTheta[1]=-currTheta[1]
        currTheta[2]=-currTheta[2]
        pose=currTheta[0:72]
        shape=currTheta[72:]
        _,j3dEst,_=smplLayer(pose.unsqueeze(0).contiguous(),shape.unsqueeze(0).contiguous())
        pelv=j3dEst[:,0,:].unsqueeze(1)
        j3dEst-=pelv
        tmp=cfg._COMMON_JOINT_IDS        
        j3dEst=j3dEst[:,tmp,:]
        regressorOut=inferModel(j3dEst.contiguous())
        plotVis.renderVisModelXyz2Smpl(r,currData,regressorOut.detach().cpu().numpy(),saveString)

if __name__ == '__main__':
    main()