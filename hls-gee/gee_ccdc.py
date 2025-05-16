#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 10:50:56 2025

@author: tvo
"""

import ee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ee.Authenticate()
ee.Initialize()

studyRegion = ee.FeatureCollection("users/anotherdayoftestingthis/paradise") # Assumes 'studyRegion' is imported (ee.Geometry or ee.FeatureCollection)
point = studyRegion.geometry().centroid()

    
startYear = 2016 # For change analysis filtering
endYear = 2024   # For change analysis filtering
startDate = str(startYear)+'-01-01' #
endDate = str(endYear)+'-12-31' 

scale = 30 #pixel size
mul_coef = 10000


# Define band names for consistency
s2BandsIn = ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12'] # BLUE, GREEN, RED, NIR, SWIR1, SWIR2
l8BandsIn = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']   # BLUE, GREEN, RED, NIR, SWIR1, SWIR2
outputBands = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']

def addNDVI(image):
    ndvi = image.normalizedDifference(['RED', 'NIR']) \
                .multiply(mul_coef) \
                .toInt16()
    return image.addBands(ndvi.rename('ndvi'))
    


#========== CLOUD MASKING FUNCTION (Clouds ONLY) ==========
#Masks ONLY pixels classified strictly as Cloud (Fmask = 4)
def mask_clouds_only(image):
    fmask = image.select('Fmask')
    
    # Keep only pixels with values of 64 or 128
    non_cloud_mask = fmask.eq(64).Or(fmask.eq(128))
    
    return (image.updateMask(non_cloud_mask)
                 .select(outputBands)
                 .copyProperties(image, ['system:time_start']))



# Get gee collection for S30 and L30
s30 = ee.ImageCollection('NASA/HLS/HLSS30/v002').filterBounds(
    studyRegion).filterDate(startDate,endDate)

                
# Apply cloud masking and rename bands
s30_masked = s30.map(lambda img: mask_clouds_only(
    img.select(
        s2BandsIn+['Fmask'],
        outputBands + ['Fmask']
    )
))
# add ndvi 
s30_masked = s30_masked.map(addNDVI)

# Multiply each band with a mul_coef
s30_masked = s30_masked.map(
    lambda image: image.multiply(mul_coef
                                 ).copyProperties(image, 
                                                   image.propertyNames())
)
        
# Load Landsat (L30) data
l30 = ee.ImageCollection('NASA/HLS/HLSL30/v002').filterBounds(
    studyRegion).filterDate(startDate, endDate)

l30_masked = l30.map(lambda img: mask_clouds_only(
    img.select(
        l8BandsIn+['Fmask'],
        outputBands + ['Fmask']
    )
))

# add ndvi 
l30_masked = l30_masked.map(addNDVI)

# Multiply each band with a mul_coef
l30_masked = l30_masked.map(
    lambda image: image.multiply(mul_coef
                                 ).copyProperties(image, 
                                                   image.propertyNames())
)


print('No of obs: S30:'+str(s30_masked.size().getInfo())+
      ' ;L30: '+str(l30_masked.size().getInfo()))
        


        
# Merge the two collections
hlsCollection = s30_masked.merge(l30_masked
                          ).sort('SYSTEM:TIME_START') # Sort by time


def normalize_with_contrast(arr, lower=2, upper=98):
    """Normalize array by clipping to given percentiles."""
    arr = arr.astype(np.float32)
    valid_mask = arr > 0
    if not np.any(valid_mask):
        return arr  # return zeros if all pixels are zero
    
    vmin = np.percentile(arr[valid_mask], lower)
    vmax = np.percentile(arr[valid_mask], upper)
    arr = np.clip(arr, vmin, vmax)
    arr = (arr - vmin) / (vmax - vmin + 1e-5)
    return arr


def plot_ee_image_rgb(image,roi,ax):
    # Sample image data as rectangle
    rgb_dict = image.sampleRectangle(region=roi, 
                                     defaultValue=0, 
                                     properties=[]).getInfo()
    # Convert to NumPy array
    r = np.array(rgb_dict['properties']['RED'], dtype=np.float32)
    g = np.array(rgb_dict['properties']['GREEN'], dtype=np.float32)
    b = np.array(rgb_dict['properties']['BLUE'], dtype=np.float32)
    
    # Convert to NumPy array
    r_norm = normalize_with_contrast(r)
    g_norm = normalize_with_contrast(g)
    b_norm = normalize_with_contrast(b)
    
    
    rgb = np.stack([r_norm, g_norm, b_norm], axis=-1)
    
    # Create mask for zero pixels across all bands
    zero_mask = (r == 0) & (g == 0) & (b == 0)

    # Set zero pixels to white
    rgb[zero_mask] = [1.0, 1.0, 1.0]
    
    # Plot
    ax.imshow(rgb)
    plt.axis('off')


fig, axes = plt.subplots(nrows=3,ncols=3)
axes = [i for ax in axes for i in ax]

# loop over each image to get RGB plot
for index in range(hlsCollection.size().getInfo()):
    
    ax = axes[index]
    
    image = ee.Image(hlsCollection.toList(hlsCollection.size()).get(index))

    plot_ee_image_rgb(image,studyRegion,ax=ax)
    






######### Step 1: Run CDCC harminized smoothing algrotihm on tiem series of SR from 
# HLS to git regression model and applied for future predictions                                                   

print('Number of images in merged collection:', hlsCollection.size().getInfo())

print('Running CCDC...')
minObservations = 8

def ccdc_temporal_segment(collection):
    ccdResults = ee.Algorithms.TemporalSegmentation.Ccdc(
        collection=collection,
        #breakpointBands=['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2'],  # Use NIR band for change detection
        breakpointBands=outputBands+['ndvi'],
        tmaskBands=['GREEN', 'SWIR2'],       # Mask using same band (optional)
        minObservations=minObservations,
        dateFormat=1
        #chiSquareProbability=0.99, # Confidence level for change detection
        #lambda_=10, # Corrected regularization parameter (default is often cited as 20, 0.002 was too low)
        #minNumOfYearsScaler= 1.33,       # Scales the minimum length of a segment
        #maxIterations=10000
    )
    print('CCDC finished. Result properties:', ccdResults.bandNames().getInfo())

    
    return ccdResults


# Convert time (ms since epoch) to datetime
def millis_to_datetime(ms):
    import datetime
    return datetime.datetime.utcfromtimestamp(ms / 1000)



def extract_pixel_value(img,point,band_name):
    
    
    return ee.Feature(point, {
        band_name: img.select(band_name).reduceRegion(ee.Reducer.first(), 
                                              point, 30).get(band_name),
        'date': img.date().format('YYYY-MM-dd')
    })
    
  


def plot_ccdc_time_series(collection,point):
    '''
    Function to plot original time series of points without harmonic fitting 

    '''
    import datetime 
    from datetime import datetime, timedelta


    # select variables for plotting   
    ccdResults = ccdc_temporal_segment(collection)
    
    ccdc_clipped = ccdResults.clip(collection.geometry().bounds())
    
    # Reduce to a single pixel at the point
    ccdc_point = ccdc_clipped.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=point,
        scale=30,
        maxPixels=1e6
    ).getInfo()
    


    
    fig, axes = plt.subplots(nrows=len(outputBands)+1)
    index=0
    for band_name in outputBands+['ndvi']:
        ax = axes[index]
        # features = collection.map(extract_pixel_value
        #                       ).filter(ee.Filter.notNull([band_name]))
        
        features = collection.map(lambda image: extract_pixel_value(image, point, band_name)
                              ).filter(ee.Filter.notNull([band_name])).sort('date')
        
        dates = features.aggregate_array('date').getInfo()
        
        dates_raw = [datetime.strptime(d, '%Y-%m-%d') for d in dates]


    
        values = features.aggregate_array(band_name).getInfo()
        
        print(len(values))
        ax.scatter(dates_raw, values, label=band_name, color='black', 
                   s=10)
        ax.set_title(band_name)

        index=index+1
        # Plot harmonic fitting using a list of coefficients
    
    return ccdc_point, dates_raw

ccdc_point, dates_raw = plot_ccdc_time_series(hlsCollection,
                                                        point,
                                                        'SWIR1')
    
    

def decimal_year_to_datetime(decimal_year):
    import datetime
    from datetime import datetime, timedelta
    
    year = int(decimal_year)
    start_of_year = datetime(year, 1, 1)
    # Days in the year (handles leap years)
    if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
        days_in_year = 366
    else:
        days_in_year = 365
    # Get the fraction of the year in days
    fractional_days = (decimal_year - year) * days_in_year
    return start_of_year + timedelta(days=fractional_days)

######### Step 2: Apply Harmonic Fitting from CCDC's coefs to fit the 
# time series of a certain band for a certain pixel #################                                                   
# Convert dates to decimal years
def to_decimal_year(d):
    import datetime
    from datetime import datetime
    
    start_of_year = datetime(d.year, 1, 1)
    end_of_year = datetime(d.year + 1, 1, 1)
    year_length = (end_of_year - start_of_year).total_seconds()
    elapsed = (d - start_of_year).total_seconds()
    return d.year + elapsed / year_length

# Harmonic fit
def harmonic_model(t, coefs):
    t = np.array(t)
    intercept = coefs[0]
    slope = coefs[1]
    fit = intercept + slope * t
    for i in range(3):  # 3 harmonics
        freq = (i + 1) * 2 * np.pi
        fit += coefs[2 + 2*i] * np.cos(freq * t)
        fit += coefs[3 + 2*i] * np.sin(freq * t)
    return fit



def harmonic_model(t, coefs):
    import math 
    t = np.array(t)
    
    PI2 = 2.0 * math.pi
    #omega = [PI2 / 365.25, PI2, PI2 / (1000 * 60 * 60 * 24 * 365.25)]
    omega = PI2

    
    return (
        coefs[0]
        + coefs[1] * t
        + coefs[2] * math.cos(t * omega)
        + coefs[3] * math.sin(t * omega)
        + coefs[4] * math.cos(t * omega * 2)
        + coefs[5] * math.sin(t * omega * 2)
        + coefs[6] * math.cos(t * omega * 3)
        + coefs[7] * math.sin(t * omega * 3)
    )




    
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
    
def plot_harmonic_fit(collection, 
                      ccdc_point, dates, band_name,
                      collection_name,
                      
                      ):
    """
    Plot original time series and harmonic fit with 8 coefficients.
    
    Parameters:
    - time_series: list of (datetime, value) tuples
    - coefs: list of 8 harmonic coefficients
    """
    import datetime
    
    features = collection.map(lambda image: extract_pixel_value(image, point, band_name)
                              ).filter(ee.Filter.notNull([band_name])).sort('date')

    dates = features.aggregate_array('date').getInfo()
            
    dates_raw = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in dates]

    
    # Get original values of time series
    original_values = features.aggregate_array(band_name).getInfo()
    
    
    
    # Get ccdc tstart, tend, and coefs 
    tstarts = [datetime.datetime(decimal_year_to_datetime(d).year,
                                 decimal_year_to_datetime(d).month,
                                 decimal_year_to_datetime(d).day
                                 )
               for d in ccdc_point['tStart']]
    
    tends = [datetime.datetime(decimal_year_to_datetime(d).year,
                                 decimal_year_to_datetime(d).month,
                                 decimal_year_to_datetime(d).day
                                 )
               for d in ccdc_point['tEnd']]
    
    coefs_list = ccdc_point[band_name+'_coefs']
    

    
    
    
    
    # Loop over each break to plot the harmonic fitting
    fig, ax = plt.subplots(figsize=(10, 5))

    index=0
    for tstart in tstarts:
        period_break = dates_raw[dates_raw.index(find_nearest(dates_raw, 
                                                           tstarts[index])):
                             dates_raw.index(find_nearest(dates_raw, 
                                              tends[index]))
                             ]
        
        decimal_years = np.array([to_decimal_year(d) for d in period_break])
        
        original_value = original_values[dates_raw.index(find_nearest(dates_raw, 
                                                           tstarts[index])):
                             dates_raw.index(find_nearest(dates_raw, 
                                              tends[index]))
                                         ]
        
        coefs = coefs_list[index]
        
        fitted = harmonic_model(decimal_years, coefs)
        
    
        # Plotting
        ax.plot(period_break, original_value, 
                'o', 
                label='Original Data', color='black')
        
        ax.plot(period_break, fitted, 
                '-', 
                label='Harmonic Fit', color='red')
    
        ax.set_title('Time Series with Harmonic Fit - '+
                     collection_name
                     )
        ax.set_xlabel('Date')
        ax.set_ylabel(band_name)
        if index == 0:
            ax.legend()
        plt.tight_layout()
        
        index=index+1



# Harmonic fitting
ccdc_point, dates_raw = plot_ccdc_time_series(hlsCollection,
                                                        point,
                                                        )

band_name = 'ndvi'
plot_harmonic_fit(hlsCollection,ccdc_point, dates_raw, band_name,
                  'S30+L30'
                  )


########## Step 3: Comparing between different collections e.g., s30, l30 
# to show the differences  in ccdc fittings #######

# S30 collection
ccdc_point, dates_raw = plot_ccdc_time_series(s30_masked,
                                                        point,
                                                        )
band_name = 'ndvi'
plot_harmonic_fit(s30_masked,ccdc_point, dates_raw, band_name,
                  'S30'
                  )


# L30 Collection
ccdc_point, dates_raw = plot_ccdc_time_series(l30_masked,
                                                        point,
                                                        )

band_name = 'ndvi'
plot_harmonic_fit(l30_masked,ccdc_point, dates_raw, band_name,
                  'L30'
                  )


    
