import torch
import numpy as np
import torch.nn as nn
import grasp_cls_pipeline_configs as configs
import matplotlib.pyplot as plt

def normalize_depth_min_max(data):
    initialized = False
    for data_point in data:
        dp_min = np.amin(data_point)
        dp_max = np.amax(data_point)
        if (dp_max-dp_min) != 0:
            data_point = (data_point-dp_min)/(dp_max-dp_min)

        if not initialized:
            new_data_points = np.array([data_point])
            initialized = True
        else:
            new_data_points = np.append(new_data_points, [data_point], 0)

    data = new_data_points
    return data

def reduce_depth_image_fidelity(depth_images, reduction_factor_x=2, reduction_factor_y=2):
    # collapse if channels is nessecary
    if len(np.shape(depth_images)) > 4:
        depth_images = np.append(depth_images[:,0,:,:], depth_images[:,1,:,:], 1)

    pool_function = nn.AvgPool2d((reduction_factor_x, reduction_factor_y), stride=(reduction_factor_x, reduction_factor_y))
    
    pooled_depth_images = pool_function(torch.from_numpy(depth_images))

    return pooled_depth_images.numpy()

def normalize_min_max(dataset):
    desired_shape = np.shape(dataset)
    min_matrix = np.broadcast_to(np.min(dataset, axis=(2,3), keepdims=True), desired_shape)
    max_matrix = np.broadcast_to(np.max(dataset, axis=(2,3), keepdims=True), desired_shape)
    result_dataset = np.divide(np.subtract(dataset,min_matrix), np.subtract(max_matrix,min_matrix)) 

    return result_dataset

def render_rgb(rgb, axsimg=None):
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.cpu().detach().numpy()

    if axsimg is None:
        rgb = prune_dimensions(rgb)
        fig, axs = plt.subplots(2)
        axsimg = [None, None, None, None]
        rgb1 = np.transpose(rgb[0:3], (1,2,0))
        rgb2 = np.transpose(rgb[3:6], (1,2,0))
        axsimg[0] = axs[0].imshow(rgb1)
        axsimg[1] = axs[1].imshow(rgb2)
    else:
        axsimg[0].set_data(rgb[0:3])
        axsimg[1].set_data(rgb[3:-1])

    plt.draw()
    plt.show(block=True)
    return axsimg

def render_depth(depth, axsimg=None):
    if axsimg is None:
        depth = prune_dimensions(depth)
        fig, axs = plt.subplots(2)
        axsimg = [None, None, None, None]
        axsimg[0] = axs[0].imshow(depth[0], cmap='gist_gray')
        axsimg[1] = axs[1].imshow(depth[1], cmap='gist_gray')
    else:
        axsimg[0].set_data(depth[0])
        axsimg[1].set_data(depth[1])

    plt.draw()
    plt.show(block=True)
    return axsimg

def prune_dimensions(array):
    if np.shape(array)[0] == 1:
        return array[0]
    else:
        return array