import os
import numpy as np

saveName="20_xyz2lspSmpl_nodropout"
rootSavePath="/home/BaseCode/trainedModels"
savePath=os.path.join(rootSavePath,saveName)
if not os.path.exists(savePath):
    os.makedirs(savePath)

inferModelName="20_xyz2lspSmpl_nodropout"
inferModelPath=os.path.join(rootSavePath,inferModelName)

# path to datasets
pathToHuman36m="/mnt/data/Human3.6M"
pathToMoSh="/mnt/data/MoSh"
pathToSurreal="/mnt/data/surreal"

debugSavePath="/mnt/data/tmp/tmp/tmp1/tmp11"

smplGender="neutral"
smplModelPath="/mnt/data/smpl"
smplMeanPath="/mnt/data/smpl/model/neutral_smpl_mean_params.h5"
#smplVisModelPath="/mnt/data/smpl/basicModel_f_lbs_10_207_0_v1.0.0.pkl"
#smplVisModelPath="/mnt/data/smpl/basicModel_m_lbs_10_207_0_v1.0.0.pkl"
smplVisModelPath="/mnt/data/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"

smplVisModelPathMale="/mnt/data/smpl/basicModel_m_lbs_10_207_0_v1.0.0.pkl"
smplVisModelPathFemale="/mnt/data/smpl/basicModel_f_lbs_10_207_0_v1.0.0.pkl"

regressorIterations=3
smplParamLossWeight=1
j3dlossWeight=1
j2dlossWeight=1 #1
encGenLossWeight=0.1
advLossWeight=0.1

ganType="lsgan"

human36mSMPLMapping=[8,5,2,1,4,7,21,19,17,16,18,20]


# params for xyz2smpl trainer
nSamplesXyz2Smpl=100
randomSeed=1
validationSplitFrac=0.1
# Mapping from SMPL 24 joints to LSP joints (0:13). In this order:
_COMMON_JOINT_IDS = [
    8,  # 8:  rightFoot ->  0: R ankle
    5,  # 5:  rightLeg -> 1: R knee
    2,  # 2:  rightUpLeg -> 2: R hip
    1,  # 1:  leftUpLeg -> 3: L hip
    4,  # 4:  leftLeg -> 4: L knee
    7,  # 7:  leftFoot -> 5: L ankle
    21, # 21: rightHand -> 6: R Wrist
    19, # 19: rightForeArm-> 7: R Elbow
    14, # 14: rightShoulder -> 8: R shoulder
    13, # 13: leftShoulder -> 9: L shoulder
    18, # 18: leftForeArm -> 10: L Elbow
    20, # 20: leftHand -> 11: L Wrist
    12, # 12  neck -> 12: # Neck top
    15, # 15: head -> 13: # Head top
]

smpl_joint_names =  [
            'hips', # 0
            'leftUpLeg', # 1
            'rightUpLeg', # 2
            'spine', # 3
            'leftLeg', # 4
            'rightLeg', # 5
            'spine1', # 6
            'leftFoot',# 7
            'rightFoot',# 8
            'spine2', #9
            'leftToeBase',# 10
            'rightToeBase',# 11
            'neck', # 12
            'leftShoulder',# 13
            'rightShoulder', # 14
            'head', # 15
            'leftArm', # 16
            'rightArm', # 17
            'leftForeArm', # 18
            'rightForeArm',# 19
            'leftHand', # 20
            'rightHand', # 21
            'leftHandIndex1', # 22
            'rightHandIndex1' # 23
            ]




nImgsLoad=1000
imgsLoad=range(nImgsLoad) #[x+10000 for x in range(10)] 
batchSize=2000
batchSizeVal=1000
learningRate=0.001 #0.00005
#weightDecay=0.001
stepSize=50000
gamma=0.9
trainIters=500000
plotSaveIters=500 # plot and save loss every plotSaveIters iters
modelSaveIters=5000 # save model every modelSaveIters iters
numVis=2


renderWidth=300
renderHeight=300
#intrinsic=np.array([[150,0,150],[0,150,150],[0,0,1]])
intrinsic=np.array([[335,0,150],[0,335,150],[0,0,1]])

normalizeFactor3d=10.0
surrealCameraSpace=False  # if True, use 3d joints in camera space, not world
surrealCropSize=224
surrealOriFocal=1200
surrealOriWidth=320
surrealOriHeight=240
surrealRenderWidth=surrealCropSize #320
surrealRenderHeight=surrealCropSize #240

surrealFx=0.5*surrealOriFocal*surrealRenderWidth/surrealOriWidth
surrealFy=0.5*surrealOriFocal*surrealRenderHeight/surrealOriHeight

surrealIntrinsic=np.array([[surrealFx,0,surrealRenderWidth/2.0],[0,surrealFy,surrealRenderHeight/2.0],[0,0,1]]) #np.array([[600,0,160],[0,600,120],[0,0,1]])
