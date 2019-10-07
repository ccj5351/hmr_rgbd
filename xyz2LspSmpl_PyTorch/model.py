import torch.nn as nn
import torchvision.models as models
import numpy as np
import torch,util
from SMPL_layer import SMPL_Layer
from sklearn.mixture import GMM
import cPickle as pickle
import config as cfg

class PosePrior(object):
    def __init__(self,pathToFile,priorType):
        if priorType=="gmm":
            f=open(pathToFile)
            self.prior=pickle.load(f)
        else:
            raise Exception("Prior type not implemented")

    def sample(self,numSamples):
        return self.prior.sample(numSamples)

class ClassBlock(nn.Module):
    def __init__(self,inpDim,outDim=512):
        super(ClassBlock, self).__init__()
        addBlock=[]
        addBlock+=[nn.Linear(inpDim,outDim)]
        addBlock+=[nn.ReLU()]
        addBlock+=[nn.Dropout(p=0.5)]
        addBlock=nn.Sequential(*addBlock)
        self.addBlock=addBlock

    def forward(self,x):
        return self.addBlock(x)

class xyz2smplRegressor(nn.Module):
    def __init__(self):
        super(xyz2smplRegressor,self).__init__()
        self.smpl=SMPL_Layer(cfg.smplGender,cfg.smplModelPath)
        self.setup()
    
    def setup(self):
        regressor=[]
        regressor+=[nn.Linear(69,64)] 
        regressor+=[nn.ReLU()]
        #regressor+=[nn.Dropout(p=0.5)]
        regressor+=[nn.Linear(64,128)] 
        regressor+=[nn.ReLU()]
        #regressor+=[nn.Dropout(p=0.5)]
        regressor+=[nn.Linear(128,128)] 
        regressor+=[nn.ReLU()]
        #regressor+=[nn.Dropout(p=0.5)]
        regressor+=[nn.Linear(128,82)] 
        self.regressor=nn.Sequential(*regressor)

    def forward(self,x):
        x=x.view(-1,x.shape[1]*x.shape[2])
        return self.regressor(x)

class xyz2lspSmplRegressor(nn.Module):
    def __init__(self):
        super(xyz2lspSmplRegressor,self).__init__()
        self.smpl=SMPL_Layer(cfg.smplGender,cfg.smplModelPath)
        self.setup()
    
    def setup(self):
        regressor=[]
        regressor+=[nn.Linear(42,64)] 
        regressor+=[nn.ReLU()]
        #regressor+=[nn.Dropout(p=0.5)]
        regressor+=[nn.Linear(64,128)] 
        regressor+=[nn.ReLU()]
        #regressor+=[nn.Dropout(p=0.5)]
        regressor+=[nn.Linear(128,128)] 
        regressor+=[nn.ReLU()]
        #regressor+=[nn.Dropout(p=0.5)]
        regressor+=[nn.Linear(128,82)] 
        self.regressor=nn.Sequential(*regressor)

    def forward(self,x):
        x=x.view(-1,x.shape[1]*x.shape[2])
        return self.regressor(x)


class PoseRegressor(nn.Module):
    def __init__(self,smplMeanPath,batchSize):
        super(PoseRegressor,self).__init__()
        self.setup(smplMeanPath,batchSize)

    def setup(self,smplMeanPath,batchSize):
        regressor=[]
        regressor+=[nn.Linear(2048,1024)] 
        regressor+=[nn.ReLU()]
        regressor+=[nn.Dropout(p=0.5)]
        regressor+=[nn.Linear(1024,1024)] 
        regressor+=[nn.ReLU()]
        regressor+=[nn.Dropout(p=0.5)]
        regressor+=[nn.Linear(1024,82)] 
        self.regressor=nn.Sequential(*regressor)
        #meanParam=np.tile(util.load_mean_theta(smplMeanPath),batchSize).reshape((batchSize,-1))
        #self.register_buffer("meanParam",torch.from_numpy(meanParam).float())

    def forward(self,x):
        params=[]
        shape=x.shape

        param=self.regressor(x)
        return param

        """param=self.meanParam[:shape[0],:]
        for r in range(cfg.regressorIterations):
            x1=torch.cat([x,param],1)
            param+=self.regressor(x1)
        return param"""

class Base(nn.Module):
    def __init__(self):
        super(Base,self).__init__()
        self.setup(cfg.smplModelPath,cfg.smplMeanPath,cfg.batchSize)
    
    def setup(self,smplModelPath,smplMeanPath,batchSize):
        model=models.resnet50(pretrained=True)
        self.featureMaps=nn.Sequential(*list(model.children())[:-2])
        self.avgPool=nn.AdaptiveAvgPool2d((1, 1))
        self.smpl=SMPL_Layer("neutral",smplModelPath)
        smplRootPos=torch.from_numpy(np.asarray(self.smpl.smpl_data["J"][0]))
        smplRootPos=smplRootPos.repeat(batchSize,1)
        self.register_buffer("smplRootPos",smplRootPos.float())
        self.smplFemale=SMPL_Layer("female",smplModelPath)
        smplRootPosFemale=torch.from_numpy(np.asarray(self.smplFemale.smpl_data["J"][0]))
        smplRootPosFemale=smplRootPosFemale.repeat(batchSize,1)
        self.register_buffer("smplRootPosFemale",smplRootPos.float())
        self.smplMale=SMPL_Layer("male",smplModelPath)
        smplRootPosMale=torch.from_numpy(np.asarray(self.smplMale.smpl_data["J"][0]))
        smplRootPosMale=smplRootPosMale.repeat(batchSize,1)
        self.register_buffer("smplRootPosMale",smplRootPos.float())
        self.regressor=PoseRegressor(smplMeanPath,batchSize)

    def forward(self,x):
        features=self.avgPool(self.featureMaps(x))
        features=features.view(features.size(0),-1)
        params=self.regressor(features)
        
        #outp=[]
        #for param in params:
        #    outp.append(self.returnOutp(param))
        #return outp

        return params

    def returnOutp(self,params):
        cam=params[:,0:3].contiguous()
        pose=params[:,3:75].contiguous()
        shape=params[:,75:].contiguous()
        #verts,j3d=self.smpl(pose,shape)
        #j2d=util.batch_orth_proj(j3d,cam)
        #return (params,verts,j2d,j3d)
        return params[:,3:]

class PoseRegressorRGBD(nn.Module):
    def __init__(self,smplMeanPath,batchSize):
        super(PoseRegressorRGBD,self).__init__()
        self.setup(smplMeanPath,batchSize)
        #self.setup1(smplMeanPath,batchSize)

    def regBlock(self,outDim):
        regressor=[]
        regressor+=[nn.Linear(1024,512)] 
        regressor+=[nn.ReLU()]
        regressor+=[nn.Dropout(p=0.5)]
        regressor+=[nn.Linear(512,256)] 
        regressor+=[nn.ReLU()]
        regressor+=[nn.Dropout(p=0.5)]
        regressor+=[nn.Linear(256,outDim)] 
        return nn.Sequential(*regressor)

    def setup(self,smplMeanPath,batchSize):
        regressor=[]
        regressor+=[nn.Linear(1024,512)] 
        regressor+=[nn.ReLU()]
        regressor+=[nn.Dropout(p=0.5)]
        regressor+=[nn.Linear(512,256)] 
        regressor+=[nn.ReLU()]
        regressor+=[nn.Dropout(p=0.5)]
        regressor+=[nn.Linear(256,85)]
        regressor+=[nn.ReLU()]
        regressor+=[nn.Linear(85,85)]
        regressor+=[nn.ReLU()]
        regressor+=[nn.Linear(85,85)]
        self.regressor=nn.Sequential(*regressor)
        meanParam=np.tile(util.load_mean_theta1(smplMeanPath),batchSize).reshape((batchSize,-1))
        self.register_buffer("meanParam",torch.from_numpy(meanParam).float())

    def setup1(self,smplMeanPath,batchSize):
        self.regressorCams=self.regBlock(3)
        self.regressorPose=self.regBlock(72)
        self.regressorShape=self.regBlock(10)
        meanParam=np.tile(util.load_mean_theta1(smplMeanPath),batchSize).reshape((batchSize,-1))
        self.register_buffer("meanParam",torch.from_numpy(meanParam).float())

    def forward(self,x):
        return self.regressor(x)

        """cams=self.regressorCams(x)
        pose=self.regressorPose(x)
        shape=self.regressorShape(x)
        inpShape=x.shape
        meanP=self.meanParam[:inpShape[0],:]
        outp=torch.cat([cams,pose,shape+meanP[:,72:]],dim=1)
        return outp"""
        
        #

class BaseRGBD_R(nn.Module):
    def __init__(self):
        super(BaseRGBD_R,self).__init__()
        self.setup()
    
    def setup(self):
        model=models.resnet50(pretrained=True)
        self.featureMaps=nn.Sequential(*list(model.children())[:-2])
        self.avgPool=nn.AdaptiveAvgPool2d((1, 1))
        self.fc=ClassBlock(2048)

    def forward(self,x):
        f=self.avgPool(self.featureMaps(x))
        f=self.fc(f.view(f.size(0),-1))
        return f

class BaseRGBD_D(nn.Module):
    def __init__(self):
        super(BaseRGBD_D,self).__init__()
        self.setup()
    
    def setup(self):
        model=models.resnet50(pretrained=True)
        self.featureMaps=nn.Sequential(*list(model.children())[:-2])
        self.avgPool=nn.AdaptiveAvgPool2d((1, 1))
        self.fc=ClassBlock(2048)

    def forward(self,x):
        f=self.avgPool(self.featureMaps(x))
        f=self.fc(f.view(f.size(0),-1))
        return f

class BaseRGBD(nn.Module):
    def __init__(self):
        super(BaseRGBD,self).__init__()
        self.setup(cfg.smplModelPath,cfg.smplMeanPath,cfg.batchSize)
    
    def setup(self,smplModelPath,smplMeanPath,batchSize):
        self.smpl=SMPL_Layer("neutral",smplModelPath)
        smplRootPos=torch.from_numpy(np.asarray(self.smpl.smpl_data["J"][0]))
        smplRootPos=smplRootPos.repeat(batchSize,1)
        self.register_buffer("smplRootPos",smplRootPos.float())
        self.smplFemale=SMPL_Layer("female",smplModelPath)
        smplRootPosFemale=torch.from_numpy(np.asarray(self.smplFemale.smpl_data["J"][0]))
        smplRootPosFemale=smplRootPosFemale.repeat(batchSize,1)
        self.register_buffer("smplRootPosFemale",smplRootPos.float())
        self.smplMale=SMPL_Layer("male",smplModelPath)
        smplRootPosMale=torch.from_numpy(np.asarray(self.smplMale.smpl_data["J"][0]))
        smplRootPosMale=smplRootPosMale.repeat(batchSize,1)
        self.register_buffer("smplRootPosMale",smplRootPos.float())
        self.regressor=PoseRegressorRGBD(smplMeanPath,batchSize)
        self.rEncoder=BaseRGBD_R()
        self.dEncoder=BaseRGBD_D()

    def forward(self,x1,x2):
        f1=self.rEncoder(x1)
        f2=self.dEncoder(x2)
        x=torch.cat((f1,f2),dim=1)
        params=self.regressor(x)
        return params

class ShapeDiscriminator(nn.Module):
    def __init__(self):
        super(ShapeDiscriminator,self).__init__()
        self.setup()

    def setup(self):
        shapeDis=[]
        shapeDis+=[nn.Linear(10,5)]
        shapeDis+=[nn.ReLU()]
        shapeDis+=[nn.Linear(5,1)]
        self.shapeDis=nn.Sequential(*shapeDis)

    def forward(self,x):
        return self.shapeDis(x)

class PoseDiscriminator(nn.Module):
    def __init__(self,channels):
        super(PoseDiscriminator,self).__init__()
        self.setup(channels)

    def setup(self,channels):
        self.convBlocks=nn.Sequential()
        l = len(channels)
        for idx in range(l-2):
            self.convBlocks.add_module(name="conv_{}".format(idx),module=nn.Conv2d(in_channels=channels[idx],out_channels=channels[idx+1],kernel_size=1,stride=1))
        self.fcLayer=nn.ModuleList()
        for idx in range(23):
            self.fcLayer.append(nn.Linear(in_features=channels[l-2],out_features=1))

    def forward(self,x):
        x=x.transpose(1,2).unsqueeze(2) # to N x 9 x 1 x 23
        convOuts=self.convBlocks(x) # to N x c x 1 x 23
        outp=[]
        for idx in range(23):
            outp.append(self.fcLayer[idx](convOuts[:,:,0,idx]))
        
        return torch.cat(outp,1),convOuts

class FullPoseDiscriminator(nn.Module):
    def __init__(self):
        super(FullPoseDiscriminator,self).__init__()
        self.setup()

    def setup(self):
        fullPoseDis=[]
        fullPoseDis+=[nn.Linear(736,1024)]
        fullPoseDis+=[nn.ReLU()]
        fullPoseDis+=[nn.Linear(1024,1024)]
        fullPoseDis+=[nn.ReLU()]
        fullPoseDis+=[nn.Linear(1024,1)]
        self.fullPoseDis=nn.Sequential(*fullPoseDis)

    def forward(self,x):
        return self.fullPoseDis(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.setup()

    def setup(self):
        self.poseDiscriminator=PoseDiscriminator([9,32,32,1])
        self.shapeDiscriminator=ShapeDiscriminator()
        self.fullPoseDiscriminator=FullPoseDiscriminator()

    def forward(self,params):
        batchSize=params.shape[0]
        if params.shape[1]==82:
            poses,shapes=params[:,:72],params[:,72:]
        else:
            poses,shapes=params[:,3:75],params[:,75:]
        shapeD=self.shapeDiscriminator(shapes)
        rotMats=util.batch_rodrigues(poses.contiguous().view(-1,3)).view(-1,24,9)[:,1:,:]
        poseD,poseDConvOuts=self.poseDiscriminator(rotMats)
        fullPoseD=self.fullPoseDiscriminator(poseDConvOuts.contiguous().view(batchSize,-1))
        #print(shapeD.shape)
        #print(poseD.shape)
        #print(fullPoseD.shape)
        return torch.cat((poseD,fullPoseD,shapeD),1)

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss





