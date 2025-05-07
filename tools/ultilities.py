#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  7 09:01:25 2025

@author: tvo
"""

def MNDWI(dataset,green_band,swir1_band):
    # Modified Normalised Difference Water Index, Xu 2006
    return (dataset[green_band] - dataset[swir1_band])/(dataset[green_band] + dataset[swir1_band])



def swir_diff(dataset,swir1_band,swir2_band):
    return dataset[swir1_band] / dataset[swir2_band]


def alpha(dataset,blue_band,green_band,swir1_band,swir2_band):
    return (2 * (np.mean(dataset[blue_band]))) / (np.mean(swir_diff(dataset,swir1_band,swir2_band)) +
                                            np.mean(MNDWI(dataset,green_band,swir1_band)**2))


def ENDISI(dataset,blue_band,green_band,swir1_band,swir2_band):
    mndwi = MNDWI(dataset,green_band,swir1_band)
    swir_diff_ds = swir_diff(dataset,swir1_band,swir2_band)
    alpha_ds = alpha(dataset,blue_band,green_band,swir1_band,swir2_band)
    
    return (dataset[blue_band] - (alpha_ds) *
            (swir_diff_ds + mndwi**2)) / (dataset[blue_band] + (alpha_ds) *
                                       (swir_diff_ds + mndwi**2))

                                          
def create_quality_mask(quality_data, bit_nums):
    """
    Uses the Fmask layer and bit numbers to create a binary mask of good pixels.
    By default, bits 1-5 are used.
    """
    mask_array = np.zeros((quality_data.shape[0], quality_data.shape[1]))
    # Remove/Mask Fill Values and Convert to Integer
    quality_data = np.nan_to_num(quality_data, 0).astype(np.int8)
    for bit in bit_nums:
        # Create a Single Binary Mask Layer
        mask_temp = np.array(quality_data) & 1 << bit > 0
        mask_array = np.logical_or(mask_array, mask_temp)
    return mask_array