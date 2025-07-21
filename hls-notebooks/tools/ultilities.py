#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Trang Vo - tv0015@uah.edu

Last modified: Mon Jul 21 11:39:04 2025

"""

def MNDWI(dataset,green_band,swir1_band):
    """
    Calculates the Modified Normalized Difference Water Index (MNDWI).

    MNDWI is used to enhance water features in remote sensing imagery,
    typically calculated as (Green - SWIR1) / (Green + SWIR1).

    Parameters
    ----------
    dataset : xarray.Dataset or similar
        Dataset containing spectral bands.
    green_band : str
        Name of the green band in the dataset.
    swir1_band : str
        Name of the short-wave infrared band 1 (SWIR1) in the dataset.

    Returns
    -------
    xarray.DataArray
        The MNDWI index values.
    """

    return (dataset[green_band] - dataset[swir1_band])/(dataset[green_band] + dataset[swir1_band])



def swir_diff(dataset,swir1_band,swir2_band):
    """
    Computes the ratio between two short-wave infrared bands (SWIR1 / SWIR2).

    This ratio can be useful for various land cover and water analyses.

    Parameters
    ----------
    dataset : xarray.Dataset or similar
        Dataset containing spectral bands.
    swir1_band : str
        Name of the first short-wave infrared band.
    swir2_band : str
        Name of the second short-wave infrared band.

    Returns
    -------
    xarray.DataArray
        The ratio of SWIR1 to SWIR2 bands.
    """

    return dataset[swir1_band] / dataset[swir2_band]


def alpha(dataset,blue_band,green_band,swir1_band,swir2_band):
    """
    Calculates the alpha parameter used in the Enhanced Normalized Difference
    Synthetic Index (ENDISI) formula.

    Alpha is computed as:
        (2 * mean(blue_band)) / (mean(swir_diff) + mean(MNDWI^2))

    Parameters
    ----------
    dataset : xarray.Dataset or similar
        Dataset containing spectral bands.
    blue_band : str
        Name of the blue band.
    green_band : str
        Name of the green band.
    swir1_band : str
        Name of the SWIR1 band.
    swir2_band : str
        Name of the SWIR2 band.

    Returns
    -------
    float
        The alpha coefficient scalar value.
    """

    return (2 * (np.mean(dataset[blue_band]))) / (np.mean(swir_diff(dataset,swir1_band,swir2_band)) +
                                            np.mean(MNDWI(dataset,green_band,swir1_band)**2))


def ENDISI(dataset,blue_band,green_band,swir1_band,swir2_band):
    """
    Computes the Enhanced Normalized Difference Synthetic Index (ENDISI).

    ENDISI is an index used for water feature enhancement and synthetic analysis, calculated as:

        (blue - alpha * (swir_diff + MNDWI^2)) / (blue + alpha * (swir_diff + MNDWI^2))

    Parameters
    ----------
    dataset : xarray.Dataset or similar
        Dataset containing spectral bands.
    blue_band : str
        Name of the blue band.
    green_band : str
        Name of the green band.
    swir1_band : str
        Name of the SWIR1 band.
    swir2_band : str
        Name of the SWIR2 band.

    Returns
    -------
    xarray.DataArray
        The ENDISI index values.
    """

    mndwi = MNDWI(dataset,green_band,swir1_band)
    swir_diff_ds = swir_diff(dataset,swir1_band,swir2_band)
    alpha_ds = alpha(dataset,blue_band,green_band,swir1_band,swir2_band)
    
    return (dataset[blue_band] - (alpha_ds) *
            (swir_diff_ds + mndwi**2)) / (dataset[blue_band] + (alpha_ds) *
                                       (swir_diff_ds + mndwi**2))
    
def create_quality_mask(quality_data, bit_nums):
    """       
    Creates a binary mask indicating pixels flagged by specified bits in a quality (Fmask) layer.

    By default, bits 1 through 5 are used if `bit_nums` is not provided.

    Parameters
    ----------
    quality_data : numpy.ndarray
        2D array of integer quality flags (e.g., from Fmask), possibly containing NaNs.
    bit_nums : list of int, optional
        List of bit positions to check in each pixel's quality flag.
        Defaults to [1, 2, 3, 4, 5].

    Returns
    -------
    numpy.ndarray
        Boolean 2D array mask where True indicates pixels flagged by any of the specified bits.

    Notes
    -----
    - NaN values in `quality_data` are replaced with zeros before processing.
    - Bit positions correspond to bits in the flag integer, with 0 being the least significant bit.
    """

    
    mask_array = np.zeros((quality_data.shape[0], quality_data.shape[1]))
    # Remove/Mask Fill Values and Convert to Integer
    quality_data = np.nan_to_num(quality_data, 0).astype(np.int8)
    for bit in bit_nums:
        # Create a Single Binary Mask Layer
        mask_temp = np.array(quality_data) & 1 << bit > 0
        mask_array = np.logical_or(mask_array, mask_temp)
    return mask_array