#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 22 May 2019

@author: M Roux
"""

#%%
import matplotlib.pyplot as plt
from skimage import data, filters
from skimage import io as skio
from scipy import ndimage
import mrlab as mr
import numpy as np

# POUR LA MORPHO
import skimage.morphology as morpho  
import skimage.feature as skf
from scipy import ndimage as ndi

#%%
def tophat(im,rayon):
    se=morpho.square(rayon)
    ero=morpho.erosion(im,se)
    dil=morpho.dilation(ero,se)
    tophat=im-dil
    return tophat
    


ima = skio.imread('spot.tif');
rayon=3
top=tophat(ima,rayon)

low = 3
high = 5


lowt = (top > low).astype(int)
hight = (top > high).astype(int)
hyst = filters.apply_hysteresis_threshold(top, low, high)
'''
fig, ax = plt.subplots(nrows=2, ncols=3)


ax[0, 0].imshow(ima, cmap='gray')
ax[0, 0].set_title('Original image')

ax[0, 1].imshow(top, cmap='magma')
ax[0, 1].set_title('Tophat filter')

ax[1, 0].imshow(lowt, cmap='magma')
ax[1, 0].set_title('Low threshold')

ax[1, 1].imshow(hight, cmap='magma')
ax[1, 1].set_title('High threshold')

ax[0, 2].imshow(hyst, cmap='magma')
ax[0, 2].set_title('Hysteresis')

ax[1, 2].imshow(hight + hyst, cmap='magma')
ax[1, 2].set_title('High + Hysteresis')

for a in ax.ravel():
    a.axis('off')

plt.tight_layout()

fig, ax = plt.subplots()
ax.imshow(top, cmap='magma')
ax.set_title('Tophat filter')
plt.show()
'''




# %%
rayons = [19, 20, 21, 22, 23, 24]
topList = []
counter = 0
for rayon in rayons:
    top = tophat(ima,rayon)

    low = 3
    high = 5

    lowt = (top > low).astype(int)
    hight = (top > high).astype(int)
    hyst = filters.apply_hysteresis_threshold(top, low, high)
    topList.append(top)
    counter = counter + 1

fig, ax = plt.subplots(nrows=2, ncols=3)
chart = [ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 0], ax[1, 1], ax[1, 2]]
counter = 0
for pos in chart:
    pos.imshow(topList[counter], cmap='magma')
    title = "Tophat filter rayon = " + str(rayons[counter])
    pos.set_title(title)
    for a in ax.ravel():
        a.axis('off')

    plt.tight_layout()
    counter = counter + 1
plt.show()
# %%
# %%
seuils = (np.arange(1, 7, 1) * 3) - 2
print(seuils)
topList = []
counter = 0
for seuil in seuils:
    top = tophat(ima,rayon)

    low = seuil
    high = seuil*2

    lowt = (top > low).astype(int)
    hight = (top > high).astype(int)
    hyst = filters.apply_hysteresis_threshold(top, low, high)
    topList.append(top)
    counter = counter + 1

fig, ax = plt.subplots(nrows=2, ncols=3)
chart = [ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 0], ax[1, 1], ax[1, 2]]
counter = 0
for pos in chart:
    pos.imshow(topList[counter], cmap='magma')
    title = "Tophat high_seuil = " + str(seuils[counter])
    pos.set_title(title)
    for a in ax.ravel():
        a.axis('off')

    plt.tight_layout()
    counter = counter + 1
plt.show()

# %%
fig, ax = plt.subplots()

ima = skio.imread('spot.tif');

rayon=3
top=tophat(ima,rayon)

low = 3
high = 4


hyst = filters.apply_hysteresis_threshold(top, low, high)
ax.imshow(hyst, cmap='magma')
ax.set_title('Hysteresis low = ' + str(low) + ' high = ' + str(high))

# %%
