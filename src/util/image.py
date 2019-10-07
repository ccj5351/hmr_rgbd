"""
Preprocessing stuff.
"""
import numpy as np
import cv2
from src.util import renderer
from src.util import surreal_in_extrinc as surreal_util


def resize_img(img, scale_factor):
    new_size = (np.floor(np.array(img.shape[0:2]) * scale_factor)).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]))
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [
        new_size[0] / float(img.shape[0]), new_size[1] / float(img.shape[1])
    ]
    return new_img, actual_factor


def scale_and_crop(image, scale, 
        center # NOTE: image center in (x,y), i.e., [width_dim, height_dim]
        , img_size):
    image_scaled, scale_factors = resize_img(image, scale)
    #print ("input image shape : ", image.shape)
    #print ("image_scaled shape : ", image_scaled.shape)
    # Swap so it's [x, y]
    scale_factors = [scale_factors[1], scale_factors[0]]
    center_scaled = np.round(center * scale_factors).astype(np.int)

    margin = int(img_size / 2)
    image_pad = np.pad(
        image_scaled, ((margin, ), (margin, ), (0, )), mode='edge')
    center_pad = center_scaled + margin
    # figure out starting point
    start_pt = center_pad - margin
    end_pt = center_pad + margin
    # crop:
    crop = image_pad[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :]
    #print ("crop  shape : ", crop.shape)
    proc_param = {
        'scale': scale,
        'start_pt': start_pt,
        'end_pt': end_pt,
        'img_size': img_size
    }

    return crop, proc_param


def scale_and_crop_with_gt(image, scale, 
        center, # NOTE: image center in (x,y), i.e., [width_dim, height_dim]
        img_size, # e.g., == 224;
        joints2d_gt, # in shape [2, 14]
        cam_gt # in shape [3,]
        ):
    image_scaled, scale_factors = resize_img(image, scale)
    print ("scale = {}, image_scaled shape : {}".format(scale, image_scaled.shape))
    # Swap so it's [x, y]
    scale_factors = [scale_factors[1], scale_factors[0]]
    center_scaled = np.round(center * scale_factors).astype(np.int)

    margin = int(img_size / 2)
    image_pad = np.pad(
        image_scaled, ((margin, ), (margin, ), (0, )), mode='edge')
    center_pad = center_scaled + margin
    # figure out starting point
    start_pt = center_pad - margin
    end_pt = center_pad + margin
    # crop:
    crop = image_pad[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :]

    # Update 2d joints;
    joints2d_gt_scaled = np.copy(joints2d_gt)
    print ("[????] joints2d_gt shape = ", joints2d_gt.shape)
    assert (joints2d_gt.shape[1] == 14)
    joints2d_gt_scaled[0,:] *= scale_factors[0] # x
    joints2d_gt_scaled[1,:] *= scale_factors[1] # y
    joints2d_gt_scaled[0,:] -= (start_pt[0] - margin) # x;
    joints2d_gt_scaled[1,:] -= (start_pt[1] - margin) # y;
    
    # Update principal point:
    cam_gt_scaled = np.copy(cam_gt)
    cam_gt_scaled[0] *= scale
    cam_gt_scaled[1] *= scale_factors[0] # tx
    cam_gt_scaled[2] *= scale_factors[1] # ty
    cam_gt_scaled[1] -= (start_pt[0] - margin) # tx
    cam_gt_scaled[2] -= (start_pt[1] - margin) # ty
    #print ("cam_gt type = {}, cam_gt = {}, cam_gt_scaled = {}".format(
    #    type(cam_gt[0]), cam_gt, cam_gt_scaled))

    print ("crop  shape : ", crop.shape)
    proc_param = {
        'scale_factors': scale_factors, # added by CCJ;
        'scale': scale,
        'start_pt': start_pt,
        'end_pt': end_pt,
        'img_size': img_size
    }

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.clf()
    plt.subplot(221)
    plt.imshow(crop[:,:,:3].astype(np.uint8))
    #print ("[crop image ***] = {}".format(crop[110:115, 110:115,:3]))
    #print ("[crop depth ***] = {}".format(crop[110:115, 110:115, 3]))
    plt.title('scaled input')
    plt.axis('off')
    
    plt.subplot(222)
    dep = crop[:,:,3]
    dep = surreal_util.normalizeDepth(dep, isNormalized = True)*255
    plt.imshow(dep.astype(np.uint8))
    plt.title('scaled depth')
    plt.axis('off')

    plt.subplot(223)
    skel_img = renderer.draw_skeleton(crop[:,:,:3].astype(np.uint8), joints2d_gt_scaled)
    plt.imshow(skel_img)
    plt.title('scaled')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(renderer.draw_skeleton(image[:,:,:3].astype(np.uint8), joints2d_gt))
    plt.title('original')
    plt.axis('off')
    plt.draw()
    plt.savefig("/home/hmr/results/surreal_debug/scale_img_joints2d_gt.png")


    return crop, proc_param, joints2d_gt_scaled, cam_gt_scaled
