#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 18:06:43 2022

@author: ckervazo
"""
import numpy as np

def gradient(M,stepX=1.,stepY=1.):
# Computes the gradient of an image, over the rows and the column directions. StepY is the assumed gap between the rows and StepX is the assumed gap between the columns   
    # Compute the x-gradient using finite differences
    gx = np.zeros_like(M)
    gx[:, 1:-1] = (M[:, 2:] - M[:, :-2]) / (2.0 * stepX)
    gx[:, 0] = (M[:, 1] - M[:, 0]) / stepX
    gx[:, -1] = (M[:, -1] - M[:, -2]) / stepX

    # Compute the y-gradient using finite differences
    gy = np.zeros_like(M)
    gy[1:-1, :] = (M[2:, :] - M[:-2, :]) / (2.0 * stepY)
    gy[0, :] = (M[1, :] - M[0, :]) / stepY
    gy[-1, :] = (M[-1, :] - M[-2, :]) / stepY

    return gx, gy
