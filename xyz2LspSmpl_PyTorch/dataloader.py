from torch.utils.data import Dataset, DataLoader
import os,glob,random,cv2,json,torch,util,cv2,h5py,math
import numpy as np
import config as cfg
from SMPL_layer import SMPL_Layer

class Surreal(Dataset):
    # Note for the surreal data
    # The 3D keyps (in world frame), after aligning
    # by pelvis, match the output of smpl(theta,beta),
    # where (theta,beta) are the smpl params given in
    # in the world frame
    def __init__(self,phase):
        self.dataPhase=phase #"train", "val", or "test"
        self.pathPrefix="/data/cmu/extracted/data/{}".format(self.dataPhase)
        self.normalize=True
        self.cropSize=cfg.surrealCropSize
        self.pixFormat="NCHW"
        self.cameraSpace=cfg.surrealCameraSpace
        self.loadData()

    def pathFilter(self,rgbP,depthP):
        s1=rgbP.split("rgb")
        s2=depthP.split("depth")
        check1=s1[0]==s2[0]
        s11=s1[1].split(".")
        s22=s2[1].split(".")
        check2=s11[0]==s22[0]
        if check1 and check2:
            return True
        else:
            return False

    def swapRightLeftPose(self,pose):
        swapInds = np.array([
                0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11, 15, 16, 17, 12, 13, 14, 18,
                19, 20, 24, 25, 26, 21, 22, 23, 27, 28, 29, 33, 34, 35, 30, 31, 32,
                36, 37, 38, 42, 43, 44, 39, 40, 41, 45, 46, 47, 51, 52, 53, 48, 49,
                50, 57, 58, 59, 54, 55, 56, 63, 64, 65, 60, 61, 62, 69, 70, 71, 66,
                67, 68
            ], np.int32)

        signFlip = np.array([   
                    1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1
                ], dtype=pose.dtype)

        newPose=np.take(pose,swapInds)*signFlip

        return newPose

    def swapRightLeftJoints(self,joints):
        assert joints.shape[1] == 24
        swapInds = np.array([0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11,10,12,14,13, 15, 17,16, 19,18,21,20,23,22], np.int32)
        jointsSwap=np.take(joints,swapInds,axis=1)
        return jointsSwap
    
    def loadData(self):
        self.keyps2d=[]
        self.keyps3d=[]
        self.smplShapeParams=[]
        self.smplPoseParams=[]
        self.rgbImages=[]
        self.depthImages=[]
        self.zrot=[]
        self.camLoc=[]
        self.gender=[]


        fp=h5py.File(cfg.pathToSurreal+self.pathPrefix+"/annot.h5","r")
        keyps2d=np.array(fp["gt2d"])
        keyps3d=np.array(fp["gt3d"])
        smplShapeParams=np.array(fp["shape"])
        smplPoseParams=np.array(fp["pose"])
        rgbImages=np.array(fp["imageNameRGB"])
        depthImages=np.array(fp["imageNameDepth"])
        camLoc=np.array(fp["camLoc"])
        zRot=np.array(fp["zRot"])
        gender=np.array(fp["gender"])
        nData=min(len(rgbImages),len(depthImages))

        for idx in range(1000): #range(nData):
            #idx=3764   #2586 #1987
            currRgbPath=rgbImages[idx]
            currDepthPath=depthImages[idx]
            ret=self.pathFilter(currRgbPath,currDepthPath)
            if ret:
                fullRgbImgPath=cfg.pathToSurreal+self.pathPrefix+currRgbPath
                fullDepthImgPath=cfg.pathToSurreal+self.pathPrefix+currDepthPath
                if (not os.path.isfile(fullRgbImgPath)) or (not os.path.isfile(fullDepthImgPath)):
                    continue
                if np.any(keyps2d[:,:,idx]<0):
                    continue
                # check if all zero depth, if so remove
                tmp=np.load(fullDepthImgPath)
                loc=tmp!=float(1e10)
                if np.min(tmp)==np.max(tmp):
                    continue
                if np.min(tmp[loc])==np.max(tmp[loc]):
                    continue
                if gender[idx][0]==1:
                    continue
                self.keyps2d.append(keyps2d[:,:,idx])
                self.keyps3d.append(keyps3d[:,:,idx])
                self.smplShapeParams.append(smplShapeParams[idx])
                self.smplPoseParams.append(smplPoseParams[idx])
                self.rgbImages.append(fullRgbImgPath)
                self.depthImages.append(fullDepthImgPath)
                self.zrot.append(zRot[idx,0])
                self.camLoc.append(camLoc[idx,:])
                self.gender.append(gender[idx])

        assert(len(self.keyps2d)==len(self.keyps3d)) 
        assert(len(self.keyps2d)==len(self.smplShapeParams)) 
        assert(len(self.keyps2d)==len(self.smplPoseParams)) 
        assert(len(self.keyps2d)==len(self.rgbImages))
        assert(len(self.keyps2d)==len(self.depthImages))
        assert(len(self.keyps2d)==len(self.zrot))
        assert(len(self.keyps2d)==len(self.camLoc))
        assert(len(self.keyps2d)==len(self.gender))

    def normalizeDepth(self,depthImg):
        loc=depthImg!=float(1e10)
        normDepth=depthImg
        normDepth[loc]=(depthImg[loc]-np.min(depthImg[loc]))/(np.max(depthImg[loc])-np.min(depthImg[loc]))
        normDepth[~loc]=0

        return normDepth

    def __len__(self):
        return len(self.rgbImages)

    def __getitem__(self,index):
        rgbImg=cv2.imread(self.rgbImages[index])
        keyps2d=self.keyps2d[index].copy()
        
        """# plot points on image and save
        imgVis=rgbImg
        for r in range(len(keyps2d[0])):
            currX=int(keyps2d[0][r])
            currY=int(keyps2d[1][r])
            cv2.circle(imgVis,(currX,currY),2,(0,0,255),4)
        cv2.imwrite("/mnt/data/tmp/tmp/tmpKeyps1.png",imgVis)"""

        keyps2d[0]=[x*(1.0*self.cropSize/rgbImg.shape[1]) for x in keyps2d[0]]
        keyps2d[1]=[x*(1.0*self.cropSize/rgbImg.shape[0]) for x in keyps2d[1]]
        keyps2d/=self.cropSize
        keyps2d=np.asarray(keyps2d)
        resImg=cv2.resize(rgbImg,(self.cropSize,self.cropSize),interpolation=cv2.INTER_CUBIC)
        resImg=util.convert_image_by_pixformat_normalize(resImg,self.pixFormat,self.normalize)
        #resImg=rgbImg

        depthImg=self.normalizeDepth(np.load(self.depthImages[index]))
        #depthImg=cv2.resize(depthImg,(self.cropSize,self.cropSize),interpolation=cv2.INTER_NEAREST)
        depthImg=cv2.resize(depthImg,(self.cropSize,self.cropSize))
        depthImgCh=np.zeros((3,depthImg.shape[0],depthImg.shape[1]))
        depthImgCh[0,:,:]=depthImg
        depthImgCh[1,:,:]=depthImg
        depthImgCh[2,:,:]=depthImg
        keyps3d=self.keyps3d[index].copy()
        keyps3d=np.asarray(keyps3d)
        keyps3dw=keyps3d.copy()

        shape,pose=self.smplShapeParams[index].copy(),self.smplPoseParams[index].copy()
        currCamLoc=self.camLoc[index].copy()
        currExtrinsic,currR,currT=util.getSurrealExtrinsic(np.expand_dims(np.transpose(currCamLoc),axis=1))
        currZRot=self.zrot[index].copy()
        RzBody=np.array(((math.cos(currZRot),-math.sin(currZRot),0),(math.sin(currZRot),math.cos(currZRot),0),(0,0,1)))        
        pose[0:3]=util.rotateBodyForVisSurreal(RzBody,pose[0:3]) 

        ############ swap l/r pose, flip y/z
        pose=self.swapRightLeftPose(pose)
        pose[1]=-pose[1]
        pose[2]=-pose[2]
        ############

        theta=np.concatenate((pose,shape),axis=0)

        ############ Swap keyps l/r
        keyps2d=self.swapRightLeftJoints(keyps2d)
        keyps3d=self.swapRightLeftJoints(keyps3d)        
        keyps3dw=keyps3d.copy()
        ############

        if self.cameraSpace:
            # move 3d joints to camera space         
            keyps3d=np.concatenate([keyps3d.transpose(),np.ones((keyps3d.shape[1],1))],axis=1).transpose()
            keyps3d=np.dot(currExtrinsic,keyps3d)   
        pelvis=keyps3d[:,0].copy() 
        keyps3d-=np.expand_dims(pelvis,1)

        currGender=self.gender[index].copy()
        currGender=currGender[0]

        return {
            "rgbImage":torch.from_numpy(resImg).float(),
            "depthImage":torch.from_numpy(depthImgCh).float(),
            "kp_2d":torch.from_numpy(keyps2d).float(),
            "kp_3d":torch.from_numpy(keyps3d).float(),
            "pelvis":torch.from_numpy(np.expand_dims(pelvis,1)).float(),
            "kp_3d_world":torch.from_numpy(np.asarray(keyps3dw)).float(),
            "theta":torch.from_numpy(theta).float(),
            "zrot":torch.from_numpy(np.asarray(currZRot)).float(),
            "camLoc":torch.from_numpy(np.asarray(currCamLoc)).float(),
            "extrinsic":torch.from_numpy(np.asarray(currExtrinsic)).float(),
            "intrinsic":torch.from_numpy(np.asarray(cfg.surrealIntrinsic)).float(),
            "rgbImageName":self.rgbImages[index],
            "depthImageName":self.depthImages[index],
            "gender":currGender,
            "data_set":"Surreal"
        }


class Human36M(Dataset):
    def __init__(self):
        self.onlySinglePerson=False
        self.normalize=True
        self.minPtsRequired=7
        self.scaleRange=[1.05,1.3]
        self.maxIntersectRatio=0.5
        self.cropSize=224
        self.pixFormat="NCHW"
        self.loadData()

    def loadData(self):
        self.keyps2d=[]
        self.keyps3d=[]
        self.smplShapeParams=[]
        self.smplPoseParams=[]
        self.images=[]
        self.boxes=[]

        fp=h5py.File(os.path.join(cfg.pathToHuman36m,"data","annot-copy.h5"),"r")
        keyps2d=np.array(fp["gt2d"])
        keyps3d=np.array(fp["gt3d"])
        smplShapeParams=np.array(fp["shape"])
        smplPoseParams=np.array(fp["pose"])
        images=np.array(fp["imagename"])

        assert(len(keyps2d)==len(keyps3d)) 
        assert(len(keyps2d)==len(smplShapeParams)) 
        assert(len(keyps2d)==len(smplPoseParams)) 
        assert(len(keyps2d)==len(images))

        def isValid(pts):
            r=[]
            for pt in pts:
                if pt[2]!=0:
                    r.append(pt)
            return r

        for idx in cfg.imgsLoad: #range(cfg.nImgsLoad): #range(len(keyps2d)): 
            fullImgPath=os.path.join(cfg.pathToHuman36m,"data")+images[idx].decode()
            if not os.path.isfile(fullImgPath):
                continue
            keyp2d=keyps2d[idx].reshape((-1,3))
            if np.sum(keyp2d[:,2])<self.minPtsRequired:
                continue
            lt,rb,v=util.calc_aabb(isValid(keyp2d))
            self.keyps2d.append(np.array(keyp2d.copy(),dtype=np.float))
            self.boxes.append((lt,rb))
            self.keyps3d.append(keyps3d[idx].copy().reshape(-1,3))
            self.smplShapeParams.append(smplShapeParams[idx].copy())
            self.smplPoseParams.append(smplPoseParams[idx].copy())
            self.images.append(fullImgPath)

    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        imgPath=self.images[index]
        keyps2d=self.keyps2d[index].copy()
        # remove head and neck points
        keyps2d=keyps2d[0:12,:]
        bbox=self.boxes[index]
        keyps3d=self.keyps3d[index].copy()
        keyps3d=keyps3d[0:12,:]

        """# plot points on image and save
        imgVis=cv2.imread(imgPath)
        for r in range(keyps2d.shape[0]):
            currX=int(keyps2d[r,0])
            currY=int(keyps2d[r,1])
            cv2.circle(imgVis,(currX,currY),2,(0,0,255),4)
            cv2.putText(imgVis,str(r),(currX,currY),cv2.FONT_HERSHEY_SIMPLEX,0.2,(255,255,255))
        cv2.imwrite("/home/BaseCode/trainedModels/5/tmpFull.png",imgVis)"""

        #scale=np.random.rand(4)*(self.scaleRange[1]-self.scaleRange[0])+self.scaleRange[0]
        scale=1
        bboxImg,bboxKps=util.cut_image(imgPath,keyps2d,scale,bbox[0],bbox[1])
        bboxKps=bboxKps[:,0:2]
        bboxKps[:,0]*=(1.0*self.cropSize/bboxImg.shape[0])
        bboxKps[:,1]*=(1.0*self.cropSize/bboxImg.shape[1])
        resImg=cv2.resize(bboxImg,(self.cropSize,self.cropSize),interpolation=cv2.INTER_CUBIC)
        #bboxKps[:,:2]=2.0*bboxKps[:, :2]*ratio-1.0
        bboxKps=(2.0*bboxKps*1.0/self.cropSize) - 1.0
        #print(bboxKps)


        shape,pose=self.smplShapeParams[index],self.smplPoseParams[index]
        #theta=np.concatenate((np.zeros(3),pose,shape),axis = 0)
        theta=np.concatenate((pose,shape),axis = 0)

        return {
            "image":torch.from_numpy(util.convert_image_by_pixformat_normalize(resImg,self.pixFormat,self.normalize)).float(),
            "kp_2d":torch.from_numpy(bboxKps).float(),
            "kp_3d":torch.from_numpy(keyps3d).float(),
            "theta":torch.from_numpy(theta).float(),
            "image_name":self.images[index],
            "w_smpl":1.0,
            "w_3d":1.0,
            "data_set":"Human3.6M"
        }

class MoSH(Dataset):
    def __init__(self):
        self.smpl=SMPL_Layer("neutral",cfg.smplModelPath)
        self.loadData()

    def loadData(self):
        fp=h5py.File(os.path.join(cfg.pathToMoSh,"data","mosh_gen","mosh_joints_annot.h5"),"r")
        self.smplShapeParams=np.array(fp["shape"])
        self.smplPoseParams=np.array(fp["pose"])
        self.smplJoints=np.array(fp["joints"])
        
        #self.smplShapeParams=self.smplShapeParams[:cfg.nSamplesXyz2Smpl,:]
        #self.smplPoseParams=self.smplPoseParams[:cfg.nSamplesXyz2Smpl,:]
        #self.smplJoints=self.smplJoints[:cfg.nSamplesXyz2Smpl,:]
        assert(len(self.smplShapeParams)==len(self.smplPoseParams)) 
        assert(len(self.smplShapeParams)==len(self.smplJoints))

    def __len__(self):
        return len(self.smplShapeParams)

    def __getitem__(self,index):
        shape,pose,joints=self.smplShapeParams[index],self.smplPoseParams[index],self.smplJoints[index]
        theta=np.concatenate((pose,shape),axis=0)
        pelv=joints[0,:]
        currXYZ=joints-pelv
        currXYZ=currXYZ[1:,:]
        return {
            "theta":torch.from_numpy(theta).float(),
            "xyz":torch.from_numpy(currXYZ).float(),
            "pelvis":torch.from_numpy(pelv).float(),
            "data_set":"MoSH"
        }



class COCO2017(Dataset):
    def __init__(self,datasetPath):
        self.onlySinglePerson=False
        self.normalize=True
        self.minPtsRequired=7
        self.scaleRange=[1.05,1.3]
        self.maxIntersectRatio=0.5
        self.cropSize=224
        self.pixFormat="NCHW"
        self.loadData(datasetPath)

    def convertToLsp14Pts(self,cocoPts):
        kpMap=[15,13,11,10,12,14,9,7,5,4,6,8,0,0]
        kpMap=[16,14,12,11,13,15,10,8,6,5,7,9,0,0]
        kps=np.array(cocoPts, dtype = np.float).reshape(-1, 3)[kpMap].copy()
        kps[12:,2]=0.0 #no neck, top head
        kps[:,2]/=2.0
        return kps
    
    def loadData(self,datasetPath):
        self.images=[]
        self.keyps=[]
        self.boxes=[]
        with open(os.path.join(datasetPath,"annotations","person_keypoints_train2017.json"),"r") as f:
            annotations=json.load(f)
        f.close()

        imgsIdInfo={}
        for inf in annotations["images"]:
            imgId=inf["id"]
            imgName=inf["file_name"]
            currAnno={}
            currAnno["image_path"]=os.path.join(datasetPath,"images","train-valid2017",imgName)
            currAnno["kps"]=[]
            currAnno["box"]=[]
            assert not (imgId in imgsIdInfo)
            imgsIdInfo[imgId]=currAnno
        
        for inf in annotations["annotations"]:
            imgId=inf["image_id"]
            kps=inf["keypoints"]
            boxInfo=inf["bbox"]
            box=[np.array([int(boxInfo[0]),int(boxInfo[1])]),np.array([int(boxInfo[0]+boxInfo[2]),int(boxInfo[1]+boxInfo[3])])]
            assert imgId in imgsIdInfo
            anno=imgsIdInfo[imgId]
            anno["box"].append(box)
            anno["kps"].append(self.convertToLsp14Pts(kps))

        self.createData(imgsIdInfo)

    def createData(self,imgsIdInfo):
        def checkCurrBboxOverlap(currBox,allOtherBoxes):
            for box in allOtherBoxes:
                if util.get_rectangle_intersect_ratio(currBox[0],currBox[1],box[0],box[1])>self.maxIntersectRatio:
                    return True
            return False
        for key,value in imgsIdInfo.items():
            currPath=value["image_path"]
            currKpsSet=value["kps"]
            currBoxSet=value["box"]
            if len(currBoxSet)>1:
                if self.onlySinglePerson:
                    continue
                
            for r in range(len(currBoxSet)):
                currKeyp=currKpsSet[r]
                currBox=currBoxSet[r]
                if np.sum(currKeyp[:,2])<self.minPtsRequired:
                    continue
                allOtherBoxes=currBoxSet.copy()
                allOtherBoxes.pop(r)
                if checkCurrBboxOverlap(currBox,allOtherBoxes):
                    continue
                self.images.append(currPath)
                self.keyps.append(currKeyp.copy())
                self.boxes.append(currBox.copy())

    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        imgPath=self.images[index]
        keyps=self.keyps[index].copy()
        bbox=self.boxes[index]

        scale = np.random.rand(4)*(self.scaleRange[1]-self.scaleRange[0])+self.scaleRange[0]
        bboxImg,bboxKps=util.cut_image(imgPath,keyps,scale,bbox[0],bbox[1])
        ratio=1.0*self.cropSize/bboxImg.shape[0]
        bboxKps[:,:2]*=ratio
        resImg=cv2.resize(bboxImg,(self.cropSize,self.cropSize),interpolation=cv2.INTER_CUBIC)
        ratio=1.0/self.cropSize
        bboxKps[:,:2]=2.0*bboxKps[:, :2]*ratio-1.0

        return {
            "image":torch.tensor(util.convert_image_by_pixformat_normalize(resImg,self.pixFormat,self.normalize)).float(),
            "kp_2d":torch.tensor(bboxKps).float(),
            "image_name":imgPath,
            "data_set":"COCO 2017"
        }

