""" This code is written based on the code Georgios 
    provided on July, 26th, 2019. 
"""

import numpy as np
import chumpy as ch
import os
import cv2
from dataloader import Human36M, Human36M_raw
#from dataloader_old import Human36M
from params import Params
import helper as hl

# Modules required for the smpl renderer
import sys
sys.path.append("/src/smpl_renderer/")
#sys.path.append("/src/smpl-pytorch-master/")
sys.path.append("/src/smpl_pavlakos/")

import numpy as np
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from smpl_webuser.serialization import load_model
#from smpl.pytorch.smpl_layer import SMPL_Layer # gulvarol smpl layer
from smpl_ import SMPL # this is pavlakos smpl layer
from pose_utils import reconstruction_error
from check_raw_data import display_model



def render_smpl(par, theta, beta, img_out_file, model_path, front_view=False):
    m = load_model(model_path)
    ## Assign the given pose
    m.pose[:] = theta
    m.betas[:] = beta
    # Define specific parameters for showing a front view of the rendering
    if front_view:
        m.pose[:3] = np.array([np.pi, 0, 0], dtype=np.float32)
        rt = np.zeros(3)
        light_source = np.array([-1000,-1000,-2000])
    else:
        rt = np.array([3.14,0,0])
        light_source = np.array([1000,1000,2000])

    ## Create OpenDR renderer
    rn = ColoredRenderer()
    ## Assign attributes to renderer
    w, h = (640, 480)
    rn.camera = ProjectPoints(v=m, rt=rt, t=np.array([0, 0, 2.]), 
             f=np.array([w,w])/2., c=np.array([w,h])/2., k=np.zeros(5))
    rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
    rn.set(v=m, f=m.f, bgcolor=np.zeros(3))

    ## Construct point light source
    rn.vc = LambertianPointLight(
        f=m.f,
        v=rn.v,
        num_verts=len(m),
        #light_pos=np.array([-1000,-1000,-2000]),
        light_pos=light_source,
        vc=np.ones_like(m)*.9,
        light_color=np.array([1., 1., 1.]))
    
    cv2.imwrite(img_out_file, rn.r*255.0)


def eval_J17(par, smpl, verts_est, gt_keypoints_3d, out_dir, visualize, img_id):
    jtr_est = smpl.get_joints(verts_est)
    gt_keypoints_3d = gt_keypoints_3d.view(par.test_batch_size, -1, 4)

    # indicator of keypoints corresponding to J17
    conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone() 
    # remove last column (conf) from gt_keypoints
    gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone() 
    conf = conf.view(-1).long() # assumes batch_size=1

    #jtr_est = jtr_est.view(par.test_batch_size, -1, 3)
    pred_pelvis = (jtr_est[:, 2, :] + jtr_est[:, 3, :]) / 2
    jtr_est = jtr_est - pred_pelvis[:, None, :]
    jtr_est = jtr_est[0, conf==1, :]
    #print(jtr_est.shape)

    verts_est = verts_est - pred_pelvis[:,None,:]    

    # taking the average between the third and fourth points in the 24 joints ??
    gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2 
    gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
    gt_keypoints_3d = gt_keypoints_3d[0, conf==1, :]
    gt_keypoints_3d[:,1] = -gt_keypoints_3d[:,1]
    gt_keypoints_3d[:,2] = -gt_keypoints_3d[:,2]
    
    #print(gt_keypoints_3d.shape)
    if visualize: # works when batch_size=1
        img_out_path = out_dir + "J17_" + img_id
        display_model(verts_est, jtr_est, img_out_path, model_faces=smpl.faces)
        #subject="S9"
        #display_model(verts_est, jtr_est, out_dir+subject+"_evalJ17_pred_centr_mesh.png", model_faces=smpl.faces)
        #display_model(verts_est, gt_keypoints_3d, out_dir+subject+"_evalJ17_gt_centr_mesh.png", model_faces=smpl.faces)

    return jtr_est, gt_keypoints_3d


def eval_J14(par, smpl, verts_est, gt_keypoints_3d, out_dir, visualize, img_id):
    # Indices to get the 14 LSP joints from the 17 H36M joints
    H36M_TO_J14 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10] 
    # Indices to get the 14 LSP joints from the ground truth joints
    J24_TO_J14 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18] 
    J_regressor_H36M_path = '/src/smpl_pavlakos/J_regressor_h36m.npy'
    J_regressor = torch.from_numpy(np.load(J_regressor_H36M_path)).float()
    J_regressor_batch = J_regressor[None, :].expand(verts_est.shape[0], -1, -1)

    jtr_est = torch.matmul(J_regressor_batch, verts_est)
    pred_pelvis = jtr_est[:, [0],:].clone()
    jtr_est = jtr_est[:, H36M_TO_J14, :]
    jtr_est = jtr_est - pred_pelvis

    jtr_est = jtr_est.view(-1, 3)
    verts_est = verts_est - pred_pelvis[:,None,:] 
    #print(jtr_est.shape)

    #gt_keypoints_3d = torch.from_numpy(gt_keypoints_3d).float()
    gt_keypoints_3d = gt_keypoints_3d.view(par.test_batch_size, -1, 4)
    # taking the average between the third and fourth points in the 24 joints ??
    gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2 
    gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
    gt_keypoints_3d = gt_keypoints_3d[:, J24_TO_J14, :-1]
    gt_keypoints_3d = gt_keypoints_3d.view(-1, 3)
    gt_keypoints_3d[:,1] = -gt_keypoints_3d[:,1]
    gt_keypoints_3d[:,2] = -gt_keypoints_3d[:,2]
    #print(gt_keypoints_3d.shape)

    if visualize:
        img_out_path = out_dir + "J14_" + img_id
        display_model(verts_est, jtr_est, img_out_path, model_faces=smpl.faces)        
        #subject="S9"
        #display_model(verts_est, jtr_est, out_dir+subject+"_evalJ14_pred_centr_mesh.png", model_faces=smpl.faces)
        #display_model(verts_est, gt_keypoints_3d, out_dir+subject+"_evalJ14_gt_centr_mesh.png", model_faces=smpl.faces)
    
    return jtr_est, gt_keypoints_3d

# in shape : N_joints x 3 
def error_metrics(jtr_est, jtr_gt):
    # Compute error metrics
    # MPJPE (mean per joint position error)
    error = torch.sqrt(((jtr_est - jtr_gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().detach().numpy()
    # Reconstruction error (MPJPE after rigid alignment of the prediction with ground truth via Procrustes Analysis)
    jtr_est = jtr_est.unsqueeze(0) # ** if batch_size > 1  then this is not needed
    jtr_gt = jtr_gt.unsqueeze(0)
    r_error = reconstruction_error(jtr_est.cpu().numpy(), jtr_gt.cpu().numpy(), reduction=None)
    return error, r_error




# Produce results
if save_res:
    res_file = open(out_dir+"res_"+par.model_id+"_"+str(par.test_iter)+"_1.txt", 'w')
    mpjpe_res_J17 = 1000 * np.asarray(mpjpe_J17).mean()
    recon_err_res_J17 = 1000 * np.asarray(recon_err_J17).mean()
    mpjpe_res_J14 = 1000 * np.asarray(mpjpe_J14).mean()
    recon_err_res_J14 = 1000 * np.asarray(recon_err_J14).mean()
    #res_file.write('Test img ids: ' + str(test_img_ids[0]) + ":" + str(test_img_ids[-1]) + "\n")
    res_file.write('Eval J17\n')
    print("Eval J17")
    res_file.write('MPJPE: ' + str(mpjpe_res_J17) + "\n")
    print('MPJPE: ' + str(mpjpe_res_J17))
    res_file.write('Reconstruction Error: ' + str(recon_err_res_J17) + "\n")
    print('Reconstruction Error: ' + str(recon_err_res_J17))
    res_file.write("\n")
    print()
    res_file.write('Eval J14\n')
    print("Eval J14")
    res_file.write('MPJPE: ' + str(mpjpe_res_J14) + "\n")
    print('MPJPE: ' + str(mpjpe_res_J14))
    res_file.write('Reconstruction Error: ' + str(recon_err_res_J14) + "\n")
    print('Reconstruction Error: ' + str(recon_err_res_J14))
    res_file.close()
