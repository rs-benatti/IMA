#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 15:22:20 2020

@author: mroux
"""
#%%

import matplotlib.pyplot as plt
from skimage import data
from skimage import io as skio
from skimage.filters import threshold_otsu
import numpy as np
import mrlab as mr
from skimage import filters

#%%
def histogram(im):
    
    nl,nc=im.shape
    
    hist=np.zeros(256)
    
    for i in range(nl):
        for j in range(nc):
            hist[im[i][j]]=hist[im[i][j]]+1
            
    for i in range(256):
        hist[i]=hist[i]/(nc*nl)
        
    return(hist)
    
    
def otsu_thresh(im, lower_mean=0, upper_mean=256):
    
    h=histogram(im)
    
    m=0
    for i in range(256):
        m=m+i*h[i]
    
    maxt=0
    maxk=0
    
    
    for t in range(256):
        w0=0
        w1=0
        m0=0
        m1=0
        for i in range(t):
            w0=w0+h[i]
            m0=m0+i*h[i]
        if w0 > 0:
            m0=m0/w0
        
        for i in range(t,256):
            w1=w1+h[i]
            m1=m1+i*h[i]
        if w1 > 0:   
            m1=m1/w1

        k=w0*w1*(m0-m1)*(m0-m1)    
        
        if k > maxk:
            maxk=k
            maxt=t
            
            
    thresh=maxt
        
    return(thresh)

def tri_class_otsu(image):
    t1 = otsu_thresh(image)
    print(t1)
    h=histogram(image)
    imageArray = np.uint32(image)
    above_t1 = imageArray[imageArray>=t1]
    below_t1 = imageArray[imageArray<t1]
    m1_lower = np.mean(below_t1)
    m1_upper = np.mean(above_t1)
    TBD = imageArray[(imageArray>= m1_lower) & (imageArray <= m1_upper)]
    t2, m2_lower, m2_upper, TBD = second_iteration_tri_class_otsu(image, m1_lower, m1_upper, TBD)
    minimunDelta = 0.01 # Stop condition: difference between thresholds
    deltaThresholds = t2 - t1
    t = []
    m_lower = []
    m_upper = []
    t.append(t2)
    m_lower.append(m2_lower)
    m_upper.append(m2_upper)
    x = 0
    while (abs(deltaThresholds) > minimunDelta):
        t_temp, m_lower_temp, m_upper_temp, TBD = second_iteration_tri_class_otsu(image, m_lower[x], m_upper[x], TBD)
        t.append(t_temp)
        m_lower.append(m_lower_temp)
        m_upper.append(m_upper_temp)
        x = x + 1
        deltaThresholds = t[x] - t[x-1]
    return t[x]
def second_iteration_tri_class_otsu(im, m1_lower, m1_upper, TBD):
    h=histogram(im)
    m1_lower = int(m1_lower)
    m1_upper = int(m1_upper)

    m=0
    for i in range(256):
        m=m+i*h[i]
    
    maxt=0
    maxk=0
    for t in range(m1_lower, m1_upper):
        w2_lower=0
        w2_upper=0
        m2_lower=0
        m2_upper=0
        for i in range(t):
            w2_lower=w2_lower+h[i]
            m2_lower=m2_lower+i*h[i]
        if w2_lower > 0:
            m2_lower=m2_lower/w2_lower
        
        for i in range(t,256):
            w2_upper=w2_upper+h[i]
            m2_upper=m2_upper+i*h[i]
        if w2_upper > 0:   
            m2_upper=m2_upper/w2_upper
        
        k=w2_lower*w2_upper*(m2_lower-m2_upper)*(m2_lower-m2_upper)    
        if k > maxk:
            maxk=k
            maxt=t
    
            
    thresh=maxt
    m2_lower = np.mean(TBD[TBD<=thresh])
    m2_upper = np.mean(TBD[TBD>=thresh])
    TBD = TBD[(TBD>=m2_lower) & (TBD <=m2_upper)]
    print(f"n2 = {thresh}")
    return(thresh, m2_lower, w2_lower, TBD)

#%%

image = skio.imread('cerveau.tif')
gfima=filters.gaussian(image,0)
grady=mr.sobelGradY(gfima)  
gradx=mr.sobelGradX(gfima)
norme=np.sqrt(gradx*gradx+grady*grady)
#image=norme 
thresh = threshold_otsu(image)
print(thresh)
binary = image > thresh

thresh=otsu_thresh(image)
binary = image > thresh
print(thresh)

thresh = tri_class_otsu(image)
binary = image > thresh
print(thresh)

'''
fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
ax = axes.ravel()
ax[0] = plt.subplot(1, 3, 1)
ax[1] = plt.subplot(1, 3, 2)
ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[0].axis('off')
'''
bins=np.max(image)-np.min(image)+1
'''
ax[1].hist(image.ravel(), bins=bins)
ax[1].set_title('Histogram')
ax[1].axvline(thresh, color='r')

ax[2].imshow(binary, cmap=plt.cm.gray)
ax[2].set_title('Thresholded')
ax[2].axis('off')
'''
fig, axes = plt.subplots(ncols=2, figsize=(8, 2.5))
ax = axes.ravel()
ax[0].imshow(binary, cmap=plt.cm.gray)
ax[0].set_title('Thresholded')
ax[1].imshow(norme, cmap=plt.cm.gray)
ax[1].set_title('Norme du gradient')
plt.show()



# %%
