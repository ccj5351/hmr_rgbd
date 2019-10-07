import torch,random,transforms3d

from datetime import datetime
import matplotlib,os,cv2,math
matplotlib.use("Agg")
from matplotlib import pyplot as plt

#from SMPL_layer import SMPL_Layer
from smpl_helpers.serialization import load_model

import numpy as np
from SMPL_layer import SMPL_Layer

import opendr
from opendr.camera import ProjectPoints
from opendr.renderer import DepthRenderer
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
import chumpy as ch

import config as cfg

class Visualizations:
    def __init__(self):
        self.savePath=cfg.savePath
        self.lossIters=[]
        self.losses=[]
        self.lossNames=[]
        #self.smpl=SMPL_Layer(cfg.smplGender,cfg.smplModelPath)
        self.model=load_model(cfg.smplVisModelPath)
        self.modelRootPos=self.model.J_transformed.r[0]
        self.smplLayer=SMPL_Layer(cfg.smplGender,cfg.smplModelPath)
        self.smplLayerRootPos=self.smplLayer.smpl_data["J"][0]
        self.dRenderer=DepthRenderer()
        self.cRenderer=ColoredRenderer()
        self.width=cfg.renderWidth 
        self.height=cfg.renderHeight
        self.numVis=cfg.numVis

    def render(self,pose,shape):
        #verts,_,faces=self.smpl(pose,th_betas=shape)
        #verts=verts.cpu().numpy()
        #faces=faces.cpu().numpy().astype(np.uint16)
        #verts=ch.array(np.squeeze(verts,axis=0))        
        self.model.pose[:]=pose
        self.model.betas[:]=shape
        verts=ch.array(self.model.r)
        faces=self.model.f
        #self.dRenderer.camera=ProjectPoints(v=verts,rt=1.0*ch.array([np.pi,0,0]),t=1*ch.array([0,0,2.0]),f=ch.array([self.width,self.width])/2.,c=ch.array([self.width,self.height])/2.,k=ch.zeros(5))
        self.dRenderer.camera=ProjectPoints(v=verts,rt=1.0*ch.array([0,0,0]),t=1*ch.array([0,0,2.0]),f=ch.array([self.width,self.width])/2.,c=ch.array([self.width,self.height])/2.,k=ch.zeros(5))
        self.dRenderer.frustum={"near":1.,"far":5.,"width":self.width,"height":self.height}
        self.dRenderer.v=verts
        self.dRenderer.f=faces
        depthArr=self.dRenderer.r.copy()
        depthArr[depthArr>4.9]=0        
        depthArr=(depthArr/5.0)*255.0
        depthArr=depthArr.astype(np.uint8)

        return depthArr

    def renderColor(self,pose,shape):       
        self.model.pose[:]=pose
        self.model.betas[:]=shape
        verts=ch.array(self.model.r)
        #vc=ch.array(self.model.vc)
        faces=self.model.f
        #self.cRenderer.camera=ProjectPoints(v=verts,rt=1.0*ch.array([np.pi,0,0]),t=1*ch.array([0,0,2.0]),f=ch.array([self.width,self.width])/2.,c=ch.array([self.width,self.height])/2.,k=ch.zeros(5))
        self.cRenderer.camera=ProjectPoints(v=verts,rt=1.0*ch.array([0,0,0]),t=1*ch.array([0,0,2.0]),f=ch.array([self.width,self.width])/2.,c=ch.array([self.width,self.height])/2.,k=ch.zeros(5))
        self.cRenderer.frustum={"near":1.,"far":10.,"width":self.width,"height":self.height}
        self.cRenderer.v=verts
        self.cRenderer.f=faces
        # Construct point light source
        self.cRenderer=LambertianPointLight(f=faces,v=verts,num_verts=len(verts),light_pos=ch.array([-1000,-1000,-2000]),vc=ch.ones_like(verts)*.9,light_color=ch.array([1., 1., 1.]))
        
        return np.fliplr(self.cRenderer.r)

    def renderHuman36(self,pose,shape,extrinsic):   
        #pose[0,0]=np.pi
        #pose[0,1]=0
        #pose[0,2]=0
        
        self.model.pose[:]=pose
        self.model.betas[:]=shape
        #self.model.trans[:]=extrinsic[0:3,3]
        verts=self.model.r
        verts=verts.copy()
        faces=self.model.f



        verts=np.concatenate([verts,np.ones((verts.shape[0],1))],axis=1).transpose()
        verts=np.dot(extrinsic,verts).transpose()

        #verts=verts+np.array([0,0,2.0])
        #print(verts)

        verts=ch.array(verts)
        #self.dRenderer.camera=ProjectPoints(v=verts,rt=1.0*ch.array([np.pi,0,0]),t=1*ch.array([0,0,0.0]),f=ch.array([335.,335.]),c=ch.array([self.width,self.height])/2.,k=ch.zeros(5))
        self.dRenderer.camera=ProjectPoints(v=verts,rt=1.0*ch.array([np.pi,0,0]),t=1*ch.array([0,0,0]),f=ch.array([320.,320.]),c=ch.array([640,480])/2.,k=ch.zeros(5))
        #self.dRenderer.camera=ProjectPoints(v=verts,rt=1.0*ch.array([np.pi,0,0]),t=1*ch.array([0.0,0.0,2.0]),f=ch.array([self.width,self.width])/2.,c=ch.array([self.width,self.height])/2.,k=ch.zeros(5))
        #self.dRenderer.camera=ProjectPoints(v=verts,rt=1.0*ch.array([0,-np.pi/2,0]),t=1*ch.array([-1.929299,0.98388302,6.74562216]),f=ch.array([1200.0,1200.0])/2.,c=ch.array([self.width,self.height])/2.,k=ch.zeros(5))
        self.dRenderer.frustum={"near":1.,"far":5.,"width":self.width,"height":self.height}
        self.dRenderer.v=verts
        self.dRenderer.f=faces
        depthArr=self.dRenderer.r.copy()
        
        depthArr[depthArr>4.9]=0        
        depthArr=(depthArr/5.0)*255.0
        depthArr=depthArr.astype(np.uint8)

        return depthArr #np.fliplr(depthArr)

    def renderSurreal(self,pose,shape,extrinsic,modelTrans=None):         

        #swap_inds=np.array([0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11, 15, 16, 17, 12, 13, 14, 18, 19, 20, 24, 25, 26, 21, 22, 23, 27, 28, 29, 33, 34, 35, 30, 31, 32, 36, 37, 38, 42, 43, 44, 39, 40, 41, 45, 46, 47, 51, 52, 53, 48, 49, 50, 57, 58, 59, 54, 55, 56, 63, 64, 65, 60, 61, 62, 69, 70, 71, 66, 67, 68])

        #sign_flip=np.array([1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1,-1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1])

        #pose=pose[swap_inds]*sign_flip

        self.model.pose[:]=pose
        self.model.betas[:]=shape
        if modelTrans is not None:
            self.model.trans[:]=modelTrans
        verts=self.model.r
        verts=verts.copy()
        faces=self.model.f

        verts=np.concatenate([verts,np.ones((verts.shape[0],1))],axis=1).transpose()
        verts=np.dot(extrinsic,verts).transpose()

        verts=ch.array(verts)
        self.dRenderer.camera=ProjectPoints(v=verts,rt=1.0*ch.array([0,0,0]),t=1*ch.array([0,0,0]),f=ch.array([cfg.surrealFx,cfg.surrealFy]),c=ch.array([cfg.surrealRenderWidth,cfg.surrealRenderHeight])/2.,k=ch.zeros(5))
        #self.dRenderer.camera=ProjectPoints(v=verts,rt=1.0*ch.array([np.pi,0,0]),t=1*ch.array([0,-1.0,2.0]),f=ch.array([1200.0,1200.0])/2.,c=ch.array([self.width,self.height])/2.,k=ch.zeros(5))
        #self.dRenderer.camera=ProjectPoints(v=verts,rt=1.0*ch.array([0,-np.pi/2,0]),t=1*ch.array([-1.929299,0.98388302,6.74562216]),f=ch.array([1200.0,1200.0])/2.,c=ch.array([self.width,self.height])/2.,k=ch.zeros(5))
        self.dRenderer.frustum={"near":1.,"far":10.,"width":cfg.surrealRenderWidth,"height":cfg.surrealRenderHeight}
        self.dRenderer.v=verts
        self.dRenderer.f=faces
        depthArr=self.dRenderer.r.copy()
        
        if np.min(depthArr)==np.max(depthArr):
            depthArrVis=np.zeros((cfg.surrealRenderWidth,cfg.surrealRenderHeight)).astype(np.uint8)
        else:
            loc=depthArr!=np.max(depthArr)
            depthArrVis=depthArr
            depthArrVis[loc]=(depthArr[loc]-np.min(depthArr[loc]))/(np.max(depthArr[loc])-np.min(depthArr[loc]))
            depthArrVis[~loc]=0
            
            depthArrVis=(depthArrVis*255).astype(np.uint8)

        return depthArrVis #np.fliplr(depthArr)

    def surrealJointNames(self):
        return ['hips',
                'leftUpLeg',
                'rightUpLeg',
                'spine',
                'leftLeg',
                'rightLeg',
                'spine1',
                'leftFoot',
                'rightFoot',
                'spine2',
                'leftToeBase',
                'rightToeBase',
                'neck',
                'leftShoulder',
                'rightShoulder',
                'head',
                'leftArm',
                'rightArm',
                'leftForeArm',
                'rightForeArm',
                'leftHand',
                'rightHand',
                'leftHandIndex1',
                'rightHandIndex1']

    def lrSwap(self,pose):
        count=0
        pose1=[]
        #swapPoints=[(1,2),(4,5),(7,8),(10,11),(13,14),(16,17),(18,19),(20,21),(22,23)]
        swapPoints=[(13,14),(16,17),(18,19),(20,21),(22,23)]
        for r in range(24):
            curr=[]
            curr.append([pose[count],pose[count+1],pose[count+2]])
            count=count+3
            pose1.append(curr)
        
        for pt in swapPoints:
            pose1[pt[0]],pose1[pt[1]]=pose1[pt[1]],pose1[pt[0]]
        flat_list = [item for sublist in pose1 for item in sublist]
        flat_list = [item for sublist in flat_list for item in sublist]

        return np.array(flat_list)
        


    def drawJoints2D(self,joints2D,ax=None,kintree_table=None,with_text=False,color='g'):
        if not ax:
            fig=plt.figure()
            ax=fig.add_subplot(111)

        for i in range(1, kintree_table.shape[1]):
            j1=kintree_table[0][i]
            j2=kintree_table[1][i]
            ax.plot([joints2D[j1,0],joints2D[j2,0]],[joints2D[j1,1],joints2D[j2,1]],color=color,linestyle='-',linewidth=1,marker='o',markersize=2)
            if with_text:
                ax.text(joints2D[j2,0],joints2D[j2,1],s=self.surrealJointNames()[j2],color=color,fontsize=8)

    def getSurrealExtrinsic(self,T):
        # Take the first 3 columns of the matrix_world in Blender and transpose.
        # This is hard-coded since all images in SURREAL use the same.
        R_world2bcam=np.array([[0,0,1],[0,-1,0],[-1,0,0]]).transpose()
        # *cam_ob.matrix_world = Matrix(((0., 0., 1, params['camera_distance']),
        #                               (0., -1, 0., -1.0),
        #                               (-1., 0., 0., 0.),
        #                               (0.0, 0.0, 0.0, 1.0)))

        # Convert camera location to translation vector used in coordinate changes
        T_world2bcam=-1*np.dot(R_world2bcam,T)

        # Following is needed to convert Blender camera to computer vision camera
        R_bcam2cv=np.array([[1,0,0],[0,-1,0],[0,0,-1]])

        # Build the coordinate transform matrix from world to computer vision camera
        R_world2cv=np.dot(R_bcam2cv,R_world2bcam)
        T_world2cv=np.dot(R_bcam2cv,T_world2bcam)

        # Put into 3x4 matrix
        RT=np.concatenate([R_world2cv,T_world2cv],axis=1)
        return RT,R_world2cv,T_world2cv

    def projectVertices(self,points,intrinsic,extrinsic):
        #extrinsic[0:3,0:3]=np.eye(3)
        homo_coords=np.concatenate([points,np.ones((points.shape[0],1))], axis=1).transpose()
        proj_coords=np.dot(intrinsic,np.dot(extrinsic,homo_coords))
        #proj_coords=np.dot(intrinsic,points)
        proj_coords=proj_coords/proj_coords[2]
        proj_coords=proj_coords[:2].transpose()
        return proj_coords

    def transformVertices(self,points,extrinsic):
        homo_coords=np.concatenate([points,np.ones((points.shape[0],1))], axis=1).transpose()
        proj_coords=np.dot(extrinsic,homo_coords)
        return proj_coords

    def weakPerspectiveProjection2d(self,X,camera,K):
        # X=K*(R*X+T), R=identity, T=trans computed below, K=intrinsic 3x3
        trans=torch.stack([camera[:,1],camera[:,2],2*cfg.surrealFx/(224*camera[:,0]+1e-9)],dim=-1).unsqueeze(1)
        X+=trans
        X=X.contiguous()
        shape=X.shape
        X=X.view(shape[0]*shape[1],shape[2]).transpose(0,1)
        Xp=torch.mm(K,X).transpose(0,1)
        Xp2=Xp[:,2].unsqueeze(-1)
        Xp=torch.div(Xp,Xp2)
        Xp=Xp.view(shape[0],shape[1],shape[2])
        Xp=Xp[:,:,:2]
        return Xp

    def denormalizeForVisSurreal(self,img,imgType):
        retImgs=[]
        if imgType=="depth":
            for r in range(img.shape[0]):
                #cImg=img[r,:,:]
                cImg=img[r,:,:,:]
                cImg=cImg.transpose(1,2,0)
                cImg=cImg[:,:,0]
                cImg=cImg*255
                cImg.astype(np.uint8)
                retImgs.append(cImg)
        else:
            for r in range(img.shape[0]):
                cImg=img[r,:,:,:]                
                cImg+=1.0
                cImg=cImg/2.0
                cImg*=255.0
                cImg=np.transpose(cImg,(1,2,0))         
                cImg.astype(np.uint8)
                retImgs.append(cImg)
        return retImgs
    
    def rotateBodyForVisSurreal(self,RzBody,pelvisRotVec):
        angle=np.linalg.norm(pelvisRotVec)
        Rpelvis=transforms3d.axangles.axangle2mat(pelvisRotVec/angle,angle)
        globRotMat=np.dot(RzBody,Rpelvis)
        R90=transforms3d.euler.euler2mat(np.pi/2,0,0)
        globRotAx,globRotAngle=transforms3d.axangles.mat2axangle(np.dot(R90,globRotMat))
        globRotVec=globRotAx*globRotAngle
        return globRotVec

    def visDataSurreal1(self,currData):
        imgsRgb=self.denormalizeForVisSurreal(currData["rgbImage"].cpu().numpy(),"rgb")
        #imgsRgb=currData["rgbImage"].cpu().numpy()
        imgsDepth=self.denormalizeForVisSurreal(currData["depthImage"].cpu().numpy(),"depth")
        keyps2d=currData["kp_2d"].cpu().numpy()
        keyps3d=currData["kp_3d"].cpu().numpy()
        theta=currData["theta"].cpu().numpy()
        extrinsic=currData["extrinsic"].cpu().numpy()

        rgbNames=currData["rgbImageName"]
        dNames=currData["depthImageName"]

        modelRootPos=self.model.J_transformed.r[0]

        for idx in range(len(imgsRgb)):
            currImg=imgsRgb[idx].copy().astype(np.uint8)

            # render curr image with params
            currKeyps2d=(keyps2d[idx,:,:]*cfg.surrealCropSize).astype(np.uint8)
            currKeyps3d=keyps3d[idx,:,:]
            currExtrinsic=extrinsic[idx,:,:] 
            if cfg.surrealCameraSpace:
                modelRootPos=np.expand_dims(modelRootPos,axis=0)
                modelRootPos=self.transformVertices(modelRootPos,currExtrinsic).transpose().squeeze(0)
                self.smplLayerRootPos=np.expand_dims(self.smplLayerRootPos,axis=0)
                self.smplLayerRootPos=self.transformVertices(self.smplLayerRootPos,currExtrinsic).transpose().squeeze(0)
            modelTrans=currKeyps3d[:,0]-modelRootPos
            currTheta=theta[idx,:]
            pose=currTheta[0:72]
            shape=currTheta[72:]            
            depthVis=self.renderSurreal(pose,shape,currExtrinsic,modelTrans)

            self.model.betas[:]=shape
            self.model.pose[:]=pose
            self.model.trans[:]=modelTrans

            smplVertices=self.model.r
            smplFaces=self.model.f
            smplJoints3D=self.model.J_transformed.r
            #print(smplJoints3D)


            smplVerticesC=smplVertices.copy()
            smplVerticesC=np.concatenate([smplVerticesC,np.ones((smplVerticesC.shape[0],1))],axis=1).transpose()
            smplVerticesC=np.dot(currExtrinsic,smplVerticesC).transpose()

            smplFaces=np.fliplr(smplFaces.copy())

            outmeshPath=os.path.join(self.savePath,"mesh{}.obj".format(idx))
            with open(outmeshPath,"w") as fp:
                for v in smplVerticesC: #self.model.r:
                    fp.write("v %f %f %f\n"%(v[0],v[1],v[2]))

                for f in smplFaces+1: # Faces are 1-based, not 0-based in obj files
                    fp.write("f %d %d %d\n"%(f[0],f[1],f[2]))

            projSmplVertices=self.projectVertices(smplVertices,cfg.surrealIntrinsic,currExtrinsic)
            projSmplJoints3D=self.projectVertices(smplJoints3D,cfg.surrealIntrinsic,currExtrinsic)

            currPoseSmplLayer=torch.from_numpy(np.expand_dims(pose,axis=0))
            currShapeSmplLayer=torch.from_numpy(np.expand_dims(shape,axis=0))
            smplLayerModelTrans=currKeyps3d[:,0]-self.smplLayerRootPos
            modelTransSmplLayer=torch.from_numpy(np.expand_dims(smplLayerModelTrans,axis=0)).float()
            _,smplLayerTestJ3d,_=self.smplLayer(currPoseSmplLayer.contiguous(),currShapeSmplLayer.contiguous(),modelTransSmplLayer.contiguous())
            smplLayerTestJ3d=smplLayerTestJ3d.detach().cpu().numpy()[0,:,:]
            #print(self.smplLayerRootPos)
            #print(smplLayerTestJ3d)
            #print(np.transpose(currKeyps3d))

            if cfg.surrealCameraSpace:
                ext=np.zeros((3,4))
                ext[0:3,0:3]=np.eye(3)
                ext[0:3,3]=np.zeros(3)
                projJoints3D=self.projectVertices(np.transpose(currKeyps3d),cfg.surrealIntrinsic,ext) 
            else:                
                projJoints3D=self.projectVertices(np.transpose(currKeyps3d),cfg.surrealIntrinsic,currExtrinsic) 
            projSmplJoints3DTestSmplLayer=self.projectVertices(smplLayerTestJ3d,cfg.surrealIntrinsic,currExtrinsic)         

            plt.clf()
            plt.subplot(3,2,1)
            plt.imshow(currImg)

            plt.subplot(3,2,2)
            plt.imshow(imgsDepth[idx])


            ax1=plt.subplot(3, 2, 3)
            plt.imshow(currImg)
            self.drawJoints2D(np.transpose(currKeyps2d),ax1,self.model.kintree_table,color='b')
            self.drawJoints2D(projSmplJoints3D,ax1,self.model.kintree_table,color='r')
            self.drawJoints2D(projJoints3D,ax1,self.model.kintree_table,color='g')

            ax1=plt.subplot(3,2,4)
            plt.imshow(currImg)
            self.drawJoints2D(np.transpose(currKeyps2d),ax1,self.model.kintree_table,color='b')
            self.drawJoints2D(projSmplJoints3D,ax1,self.model.kintree_table,color='r')
            self.drawJoints2D(projSmplJoints3DTestSmplLayer,ax1,self.model.kintree_table,color='k')

            plt.subplot(3, 2, 5)
            plt.imshow(currImg)
            plt.scatter(projSmplVertices[:,0],projSmplVertices[:,1],0.01)
            plt.savefig(os.path.join(self.savePath,"testVis{}.png".format(idx)))

            plt.subplot(3, 2, 6)
            plt.imshow(cv2.cvtColor(depthVis,cv2.COLOR_GRAY2RGB))
            plt.savefig(os.path.join(self.savePath,"testVis{}.png".format(idx)))

    def visDataHuman36(self,currData):
        imgsPath=currData["image_name"]
        gtParams=currData["theta"].cpu().numpy()
        keyps3d=currData["kp_3d"].cpu().numpy()
        nImgs=len(imgsPath)

        R=np.array([[-0.90420742,0.06390494,-0.42228557],[0.42657831,0.18368565, -0.88560179],[0.02097347,-0.98090557,-0.19335039]])
        #R=np.linalg.inv(R)
        #R=np.array([[0.9222116,0.09333184,-0.37525316],[0.38649076,-0.19167234,0.90215664],[0.01227429,-0.9770112,-0.21283435]])
        #R=np.array([[-0.92582886,-0.02357811,0.37720683],[-0.37286741,0.22000056,0.90142645],[-0.06173178,-0.97521476,-0.21247438]])
        #R=np.array([[0.92228155,-0.02117765,0.38593814],[-0.37726887,-0.26645871,0.88694304],[0.08405321,-0.96361365,-0.25373962]])
        

        #R_bcam2cv=np.array([[1,0,0],[0,0,-1],[0,-1,0]])
        #R=np.dot(R_bcam2cv,R)

        
        T=np.array([[2097.39151027],[4880.94465755],[1605.73247184]])/1000.0
        #T=np.array([[2031.70078497],[-5167.93301207],[1612.92305082]])/1000.0
        #T=np.array([[-1620.59486279],[5171.65873305],[1496.43704697]])/1000.0
        #T=np.array([[-1637.17374541],[-3867.31734917],[1547.03325639]])/1000.0

        #T=np.array([[0],[0.],[3.]])

        
        #T=np.dot(R_bcam2cv,T)
        
        #R=cv2.Rodrigues(np.array([3.14,0,0]))[0]
        #T=np.array([[0],[0],[2.]])
        #T=np.array([[0],[0.],[3.]])
        


        currExtrinsic=np.concatenate([R,T],axis=1)
        #print(currExtrinsic)
        #currExtrinsic=np.concatenate([np.eye(3),np.zeros((3,1))],axis=1)
       
        for r in range(nImgs):
            currTestImg=cv2.cvtColor(cv2.imread(imgsPath[r]),cv2.COLOR_RGB2GRAY)
            currPoseGt=np.expand_dims(gtParams[r,0:72],axis=0)
            currPoseGt[0,1]=-currPoseGt[0,1]
            currShapeGt=np.expand_dims(gtParams[r,72:],axis=0)
            currKeyps3d=keyps3d[r,:,:]

            #print(currKeyps3d*1000)

            #rootOri=currPoseGt[:,0:3]
            #print(rootOri.shape)
            #rootOri[0,1]-=(np.pi/2)
            #rootOri[0,1]=-rootOri[0,1]
            #rootOri[0,2]=np.pi-rootOri[0,2]
            #R=cv2.Rodrigues(rootOri)[0]   
            #R_bcam2cv=np.array([[-1,0,0],[0,-1,0],[0,0,-1]])
            #R_bcam2cv=np.array([[0,1,0],[1,0,0],[0,0,1]])
            #R=np.dot(R_bcam2cv,R)
            #currExtrinsic=np.concatenate([R,T],axis=1)

            #currPoseGt=np.zeros((1,72))
            #currPoseGt[0,0]=0
            #currPoseGt[0,1]=0
            #currPoseGt[0,2]=0

            depthVis=self.renderHuman36(currPoseGt,currShapeGt,currExtrinsic)
            #saveImg=np.hstack((currTestImg,depthVis))
            #cv2.imwrite(os.path.join(self.savePath,"testVis{}.png".format(r)),saveImg)
            cv2.imwrite(os.path.join(self.savePath,"testVisImg{}.png".format(1)),currTestImg)
            cv2.imwrite(os.path.join(self.savePath,"testVisRend{}.png".format(1)),depthVis)

            print(currPoseGt[0,0:3])

            """self.model.betas[:]=currShapeGt
            self.model.pose[:]=currPoseGt

            smplVertices=self.model.r
            smplFaces=self.model.f
            smplJoints3D=self.model.J_transformed.r

            print(smplJoints3D)

            projSmplVertices=self.projectVertices(smplVertices,cfg.intrinsic,currExtrinsic)
            projJoints3D=self.projectVertices(currKeyps3d,cfg.intrinsic,currExtrinsic) 

            plt.clf()
            ax1=plt.subplot(1,2,1)
            plt.imshow(currTestImg)
            plt.scatter(projSmplVertices[:,0],projSmplVertices[:,1],0.01)

            plt.subplot(1,2,2)
            plt.imshow(currTestImg)
            plt.scatter(projJoints3D[:,0],projJoints3D[:,1],0.01)
            plt.savefig(os.path.join(self.savePath,"testVisScat{}.png".format(r)))

            smplVerticesC=smplVertices.copy()
            smplVerticesC=np.concatenate([smplVerticesC,np.ones((smplVerticesC.shape[0],1))],axis=1).transpose()
            smplVerticesC=np.dot(currExtrinsic,smplVerticesC).transpose()

            outmeshPath=os.path.join(self.savePath,"mesh{}.obj".format(r))
            with open(outmeshPath,"w") as fp:
                for v in smplVerticesC: #self.model.r:
                    fp.write("v %f %f %f\n"%(v[0],v[1],v[2]))

                for f in smplFaces+1: # Faces are 1-based, not 0-based in obj files
                    fp.write("f %d %d %d\n"%(f[0],f[1],f[2]))"""


    def updatePlotLoss(self,lossDict,plot=False):

        itemsList=lossDict.items()
        self.lossIters.append(itemsList[0][1])
        if not self.losses:
            for r in range(len(itemsList)-1):
                self.losses.append([])
                self.lossNames.append(itemsList[r+1][0])
        for r in range(len(itemsList)-1):
            self.losses[r].append(itemsList[r+1][1])

        if plot:
            for r in range(len(self.losses)):
                plt.plot(self.lossIters[40:],self.losses[r][40:],linewidth=2.0)
                plt.xlabel("Iterations",fontsize=14)
                plt.ylabel("{}".format(self.lossNames[r]),fontsize=14)
                if not os.path.exists(self.savePath):
                    os.makedirs(self.savePath)
                plt.savefig(os.path.join(self.savePath,"{}.png".format(self.lossNames[r])))
                plt.close("all")

    def renderVisModel(self,iterIndex,testImgsPath,gtParams,estParams,savePathString):
        nImgs=len(testImgsPath)
        showVis=random.sample(range(nImgs),self.numVis)
        savePath=os.path.join(self.savePath,savePathString)
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        for r in showVis: #range(nImgs):
            currTestImg=cv2.resize(cv2.cvtColor(cv2.imread(testImgsPath[r]),cv2.COLOR_BGR2GRAY),(self.width,self.height))
            #estParams[r,0]=np.pi
            #estParams[r,1:3]=0
            currPose=torch.from_numpy(np.expand_dims(estParams[r,0:72],axis=0))
            currShape=torch.from_numpy(np.expand_dims(estParams[r,72:],axis=0))

            depthVis=self.render(currPose,currShape)

            #gtParams[r,0]=np.pi
            currPoseGt=torch.from_numpy(np.expand_dims(gtParams[r,0:72],axis=0))
            currShapeGt=torch.from_numpy(np.expand_dims(gtParams[r,72:],axis=0))

            depthVisGt=self.render(currPoseGt,currShapeGt)
            
            saveImg=np.hstack((currTestImg,depthVis,depthVisGt))

            cv2.imwrite(os.path.join(savePath,"iter{}_{}.png".format(iterIndex,r)),saveImg)

    def renderVisModelSurreal(self,iterIndex,currData3d,estParams,savePathString):
        testImgsPath=currData3d["rgbImageName"]
        nImgs=len(testImgsPath)
        gtParams=currData3d["theta"].cpu().numpy()
        extrinsic=currData3d["extrinsic"].cpu().numpy()
        intrinsic=currData3d["intrinsic"].cpu().numpy()
        keyps3d=currData3d["kp_3d"].cpu().numpy()
        showVis=random.sample(range(nImgs),self.numVis)
        savePath=os.path.join(self.savePath,savePathString)
        if not os.path.exists(savePath):
            os.makedirs(savePath)        
        for r in showVis: 
            currTestImg=cv2.resize(cv2.cvtColor(cv2.imread(testImgsPath[r]),cv2.COLOR_BGR2GRAY),(cfg.surrealCropSize,cfg.surrealCropSize))
            currExtrinsic=extrinsic[r,:,:]
            currIntrinsic=intrinsic[r,:,:]
            currKeyps3d=keyps3d[r,:,:]
            modelTrans=currKeyps3d[:,0]-self.modelRootPos
            currPose=estParams[r,0:72]
            currShape=estParams[r,72:]

            depthVis=self.renderSurreal(currPose,currShape,currExtrinsic,modelTrans)

            currPoseGt=gtParams[r,0:72]
            currShapeGt=gtParams[r,72:]

            depthVisGt=self.renderSurreal(currPoseGt,currShapeGt,currExtrinsic,modelTrans)
            
            saveImg=np.hstack((currTestImg,depthVis,depthVisGt))

            cv2.imwrite(os.path.join(savePath,"iter{}_{}.png".format(iterIndex,r)),saveImg)

    def renderVisModelSurreal2(self,iterIndex,currData3d,estParams,savePathString):
        testImgsPath=currData3d["rgbImageName"]
        nImgs=len(testImgsPath)
        gtParams=currData3d["theta"].cpu().numpy()
        intrinsic=currData3d["intrinsic"].cpu().numpy()
        extrinsic=currData3d["extrinsic"].cpu().numpy()
        keyps2d=currData3d["kp_2d"].cpu().numpy()
        keyps3d=currData3d["kp_3d"].cpu().numpy()
        pelvis=currData3d["pelvis"].cpu().numpy()
        keyps3dw=currData3d["kp_3d_world"].cpu().numpy()
        showVis=random.sample(range(nImgs),self.numVis)
        savePath=os.path.join(self.savePath,savePathString)
        #savePath=os.path.join(cfg.inferModelPath,savePathString)
        if not os.path.exists(savePath):
            os.makedirs(savePath)        
        for r in showVis: 
            currTestImg=cv2.resize(cv2.imread(testImgsPath[r]),(cfg.surrealCropSize,cfg.surrealCropSize))
            currIntrinsic=intrinsic[0,:,:]
            currExtrinsicGt=extrinsic[r,:,:]
            currKeyps2d=(keyps2d[r,:,:]*cfg.surrealCropSize).astype(np.uint8)
            #currKeyps2d=((keyps2d[r,:,:]+1.0)*cfg.surrealCropSize/2.0).astype(np.uint8)
            currKeyps3d=keyps3d[r,:,:]+pelvis[r,:,:]
            modelTrans=currKeyps3d[:,0]-self.modelRootPos

            currPoseGt=gtParams[r,0:72]
            #currPoseGt[0]=np.pi
            #currPoseGt[1]=0
            #currPoseGt[2]=0
            currShapeGt=gtParams[r,72:]

            currCam=estParams[r,0:3]
            currPose=estParams[r,3:75]
            #currPose[0]=np.pi
            #currPose[1]=0
            #currPose[2]=0
            currShape=estParams[r,75:]

            depthVisGt=self.renderSurreal(currPoseGt,currShapeGt,currExtrinsicGt,modelTrans)

            currExtrinsicEst=currExtrinsicGt.copy()
            currExtrinsicEst[0:3,0:3]=np.eye(3)
            currExtrinsicEst[0:2,3]=currCam[1:3]
            currExtrinsicEst[2,3]=2*cfg.surrealFx/(cfg.surrealCropSize*currCam[0]+1e-9)
            depthVis=self.renderSurreal(currPose,currShape,currExtrinsicEst)

            # Project joints on image

            # First, gt 
            if cfg.surrealCameraSpace:
                gt3dextrinsic=np.zeros((3,4))
                gt3dextrinsic[0:3,0:3]=np.eye(3)
                projJoints3D=self.projectVertices(np.transpose(currKeyps3d),cfg.surrealIntrinsic,gt3dextrinsic)
            else:
                projJoints3D=self.projectVertices(np.transpose(currKeyps3d),cfg.surrealIntrinsic,currExtrinsicGt)

            # Next, project predicted joints

            currPoseSmplLayer=torch.from_numpy(np.expand_dims(currPose,axis=0))
            currShapeSmplLayer=torch.from_numpy(np.expand_dims(currShape,axis=0))
            #trans=np.array([currCam[1],currCam[2],2*cfg.surrealFx/(cfg.surrealCropSize*currCam[0]+1e-9)])
            #smplLayerModelTrans=self.smplLayerRootPos+trans
            #smplLayerModelTrans=torch.from_numpy(np.expand_dims(smplLayerModelTrans,axis=0)).float()
            _,smplLayerTestJ3d,_=self.smplLayer(currPoseSmplLayer.contiguous(),currShapeSmplLayer.contiguous())
            currCams=torch.from_numpy(np.expand_dims(currCam,axis=0))
            smplLayerTestJ2d=self.weakPerspectiveProjection2d(smplLayerTestJ3d,currCams,torch.from_numpy(currIntrinsic))
            smplLayerTestJ2d=smplLayerTestJ2d[0,:,:].detach().cpu().numpy()


            plt.clf()
            plt.subplot(2,3,1)
            plt.imshow(currTestImg)

            plt.subplot(2,3,2)
            plt.imshow(depthVisGt)

            plt.subplot(2,3,3)
            plt.imshow(depthVis)

            ax1=plt.subplot(2,3,4)
            plt.imshow(currTestImg)
            self.drawJoints2D(np.transpose(currKeyps2d),ax1,self.model.kintree_table,color='r')

            ax1=plt.subplot(2,3,5)
            plt.imshow(currTestImg)
            self.drawJoints2D(projJoints3D,ax1,self.model.kintree_table,color='g')
            #plt.savefig(os.path.join(savePath,"testVis{}.png".format(r)))

            ax1=plt.subplot(2,3,6)
            plt.imshow(currTestImg)
            self.drawJoints2D(smplLayerTestJ2d,ax1,self.model.kintree_table,color='b')
            plt.savefig(os.path.join(savePath,"testVis{}_{}.png".format(iterIndex,r)))

            plt.close("all")

    def renderVisModelXyz2Smpl(self,iterIndex,currData3d,estParams,savePathString):
        gtParams=currData3d["theta"].cpu().numpy()
        #gtXyz=currData3d["xyz"].cpu().numpy()
        #gtPelv=currData3d["pelvis"].cpu().numpy()
        showVis=random.sample(range(gtParams.shape[0]),self.numVis)
        savePath=os.path.join(self.savePath,savePathString)

        if not os.path.exists(savePath):
            os.makedirs(savePath)        
        for r in range(len(showVis)): 
            currGtParams=gtParams[r,:]
            currEstParams=estParams[r,:]

            gtPose=currGtParams[:72].copy()
            estPose=currEstParams[:72].copy()
            gtPose[0]=np.pi
            gtPose[1]=0
            gtPose[2]=0
            estPose[0]=np.pi
            estPose[1]=0
            estPose[2]=0
            depthVisGt=self.render(gtPose,currGtParams[72:])
            depthVisEst=self.render(estPose,currEstParams[72:])

            currSavePath=os.path.join(savePath,"testVis{}_{}.png".format(iterIndex,r))
            plt.clf()
            plt.subplot(1,2,1)
            plt.imshow(depthVisGt)
            plt.subplot(1,2,2)
            plt.imshow(depthVisEst)
            plt.savefig(currSavePath)
            plt.close("all")



