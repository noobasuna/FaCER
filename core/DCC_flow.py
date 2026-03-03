#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 10:22:15 2022

@author: s434626
"""
import torch
import torch.nn.functional as F
import numpy as np
import os
import yaml
import copy
from PIL import Image
from os import path as osp

def save_imgs_as_arrays(img, denorm_fn=lambda x: x * 0.5 + 0.5):
    img = denorm_fn(img.detach().cpu().numpy())
    img = np.transpose(img, axes=(0, 2, 3, 1))
    img = (img * 255).astype('uint8')
    
    image_arrays = []

    for idx, i in enumerate(img):
        i = Image.fromarray(i)
        np.save(f'{idx}.npy',i)
        
        
def save_imgs_as_arrays_1(img, denorm_fn=lambda x: x * 0.5 + 0.5):
    img = denorm_fn(img.detach().cpu().numpy())
    img = np.transpose(img, axes=(0, 2, 3, 1))
    img = (img * 255).astype('uint8')
    
    image_arrays = []

    for idx, i in enumerate(img):
        i = Image.fromarray(i)
        np.save(f'{idx}_1.npy',i)
        
def differential_covariance(X, eps=1e-12, alpha=1e-3):
    """ uses ridge regularization 
    """
    import numpy as np 
    
    # standardize
    # X_ = (X - np.nanmean(X, axis=1)[:,None])  / ( np.nanstd(X, axis=1)[:,None] + eps )
    X_ = X.copy()
    
    # # differential 
    # X_pad = np.pad(X_, pad_width=[[0,0],[1,1]], mode='edge')
    # dX_ = (X_pad[:,2:] - X_pad[:,:-2]) / 2.
    # # dX_ = np.gradient(X_, axis=1)
    dX_ = X_[:,1:] - X_[:,:-1]
    X_ = X_[:,1:].copy()
    # print(X_.shape,dX_.shape)
    X_ = X_.T
    dX_ = dX_.T
    
    # linear least squares solution . 
    dX_X = dX_.T.dot(X_)
    X_X = X_.T.dot(X_)
    
    # W = np.linalg.solve(X_X+reg*np.eye(len(X_X)), dX_X) # transpose... 
    W = dX_X.dot(np.linalg.inv(X_X +alpha*np.eye(len(X_X))))
    
    return W 


# def DDC_cause(img1, img2, dilation,
#                 k=1, m=3, 
#                 eta_xt=5e-4, 
#                 eta_yt=5e-4,
#                 eta_xtkm=5e-4):
    
#     save_imgs_as_arrays(img1)
#     save_imgs_as_arrays_1(img2)
#     for number in range(0, 3):
#     """
#     img1 : (M,N,T) array
#     img2 : (M,N,T) array

# """
#         img1 = np.load(f'{number}.npy')
#         img2 = np.load(f'{number}_1.npy')
#         Y = img1.copy() #- np.mean(img1)
#         X = img2.copy() #- np.mean(img1)

#         Y = Y.reshape(-1, Y.shape[-1]).T
#         X = X.reshape(-1, X.shape[-1]).T

#         calc_xy = PCCA_GC_Calculator(X=X, Y_cause=Y)
#         # Gy_to_x = calc_xy.calcGrangerCausality(k=1, m=1,
#         #                                        eta_xt=1e-5, eta_yt=1e-5, eta_xtkm=1e-5) # delay lag=1 and order=1 # this is slow.... 
#         Gy_to_x = calc_xy.calcGrangerCausality(k=k, m=m,
#                                                eta_xt=eta_xt, 
#                                                eta_yt=eta_yt, 
#                                                eta_xtkm=eta_xtkm) # etas are very important 
#     return Gy_to_x


def DDC_cause(img1, img2, dilation, eps=1e-12, alpha=1e-2):
    
    import numpy as np 
    import pylab as plt 

    save_imgs_as_arrays(img1)
    save_imgs_as_arrays_1(img2)
    corr_arrays = []
    
    for number in range(0, 3):
        img1 = np.load(f'{number}.npy')
        img2 = np.load(f'{number}_1.npy')
        print('mask1 done')
        # compile all the timeseries
        Y_ = np.array([img1, 
                       img2])
        Y_ = Y_.reshape(len(Y_), -1, Y_.shape[-1])

        """
        Compute the diff covariance 
        """
        W_ = differential_covariance(Y_.reshape(-1, Y_.shape[-1]), eps=eps, alpha=alpha)

        # W_ = W_.T.copy()
        N = Y_.shape[1] # this is the flattened over spatial windows. 
        N_rows = int(np.sqrt(N))

        # W_out = W_[1:,0].copy() # - W_[0,1:]
        W_out = W_[:N, N:].copy()
        corr_array = np.reshape(W_out,axes=(3,3,128,128))
#         corr_array = np.zeros((N_rows,N_rows))

#         for ii in np.arange(N_rows):
#             for jj in np.arange(N_rows):
#                 ind = ii*N_rows + jj
#                 corr_array[ii,jj] = W_out[ind,ind]
        corr_arrays.append(corr_array)
        print(corr_arrays.shape)
    
    corr_arrays = np.transpose(corr_arrays, axes=(0, 3, 1, 2)).sum(dim=1, keepdim=True)
    corr_array = corr_array.reshape((winsize,winsize))
    mid = corr_array.shape[1]//2

    corr_x_direction = -np.nansum(corr_array[:,:mid]) + np.nansum(corr_array[:,mid+1:])
    corr_y_direction = -np.nansum(corr_array[:mid]) + np.nansum(corr_array[mid+1:])
    intensity = np.nansum(corr_array) #* np.sqrt(corr_x_direction**2 + corr_y_direction**2)

    mean_vector = np.hstack([corr_y_direction, corr_x_direction])
    mean_vector = mean_vector * intensity
    
    # mask = torch.from_numpy(corr_arrays)
    # dil_mask = F.max_pool2d(mask,
    #                     dilation, stride=1,
    #                     padding=(dilation - 1) // 2)
    
    return corr_array, dil_mask
    