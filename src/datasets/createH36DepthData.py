import numpy as np
import cv2,os
from glob import glob
from spacepy import pycdf

srcDir="/mnt/ccj/S5/TOF"
dstDir="/mnt/ccj/S5/DepthImages/"
cdfPaths=glob(os.path.join(srcDir,"*.cdf"))
maxDepthNorm=15.0

for currCdf in cdfPaths:
    currData=pycdf.CDF(currCdf)
    currSaveDir=os.path.join(dstDir,os.path.splitext(os.path.basename(currCdf))[0])
    if not os.path.exists(currSaveDir):
        os.makedirs(currSaveDir)
    allRFrames=currData["RangeFrames"]
    allRFrames=np.squeeze(allRFrames,axis=0)
    for r in range(allRFrames.shape[2]):
        currRange=(allRFrames[:,:,r])
        currRangeVis=currRange.copy()
        cv2.imwrite(os.path.join(currSaveDir,"{}.exr".format(r)),currRange.astype(np.float32))
        currRangeVis=currRangeVis*255/maxDepthNorm
        cv2.imwrite(os.path.join(currSaveDir,"{}_vis.png".format(r)),currRangeVis.astype(np.uint8))
