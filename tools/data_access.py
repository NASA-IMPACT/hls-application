#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  7 08:58:57 2025

@author: tvo
"""

import earthaccess
import pandas as pd
import numpy as np
import pystac_client
import odc.stac
import xarray as xr
import rasterio
from rasterio.env import Env
import os

def get_HLS_data(lat_range,lon_range,baseline_year,analysis_year):    
    temporal = (str(baseline_year)+"-01-01",str(analysis_year)+"-12-31")
    
    results = earthaccess.search_data(
    short_name=['HLSL30','HLSS30'],
    bounding_box=(min(lon_range),min(lat_range),max(lon_range),max(lat_range)),
    temporal=temporal,
    )
    
    df = pd.json_normalize(results)

    return df, results
    
def filter_items_by_id(item, keyword):
    return keyword in item.id

    
def search_cmr_stac(baseline_year,analysis_year,lat_range,lon_range,date_begin=None,date_end=None,granule_id=None):
    bbox = ([min(lon_range),min(lat_range),max(lon_range),max(lat_range)])
    if type(baseline_year) is int:
        years = np.arange(baseline_year,analysis_year+1,1)
    else:
        years = np.arange(int(baseline_year.split('-')[0]),
                          int(analysis_year.split('-')[0])+1
                          ,1)
    
    # Due to the limitation of the search, split into every year to perform the search
    # Note that this might work for a city scale, but for larger domain, the search can
    # even further needes to be constrained 
    items_list = list()
    for year in years:
        print(year)
        collections=['HLSL30.v2.0', 'HLSS30.v2.0']
        url='https://cmr.earthdata.nasa.gov/cloudstac/LPCLOUD'
        cloudcover_max=50
        lim=100
        if date_begin is not None:
            dt_min = str(year)+'-'+date_begin
            dt_max = str(year)+'-'+date_end
        else:
            dt_min = str(year)+'-01-01'
            dt_max = str(year)+'-12-31'
            
        # open the catalog
        catalog = pystac_client.Client.open(f'{url}')
        
        # perform the search
        search = catalog.search(
            collections=collections,
            bbox=bbox,
            datetime=dt_min + '/' + dt_max,
            limit=lim
        )
    
        items = list(search.items())

        if granule_id is not None:
            items = [item for item in items if granule_id in item.id]
                    
        
        items_list.append(items)
        print('Found', len(items), 'granules at point', bbox, 'from', dt_min, 'to', dt_max)

    
    items_list = [i for item in items_list for i in item]

    return items_list


def rename_common_bands(items_list):
    # Rename HLSS B8A and HLSL B05 to common band name
    S30_band_common = ['B02','B03','B04','B8A','B11','B12','B10']
    L30_band_common = ['B02','B03','B04','B05','B06','B07','B09']
    band_name = ['Blue','Green','Red','NIR','SWIR_1','SWIR_2','Cirrus']
    index=0
    for band in S30_band_common:
        
        for item in items_list:
            if "HLS.L30" in item.id:
                item.assets[band_name[index]] = item.assets.pop(L30_band_common[index])
            if "HLS.S30" in item.id:
                item.assets[band_name[index]] = item.assets.pop(band)
        
        index=index+1
            
    return items_list

def load_odc_stac(crs,bands,spatial_res,items_list,bbox):

    # Set CRS and resolution, open lazily with odc.stac
    ds = odc.stac.stac_load(
        items_list,
        bbox=bbox,
        bands=(bands),
        crs=crs,
        resolution=spatial_res,
        #chunks=None,
        chunks={"band":1,"x":512,"y":512},  # If empty, chunks along band dim, 
        #groupby="solar_day", # This limits to first obs per day
    )

    return ds
    

def load_dask(ds):

    import rasterio
    gdal_cookiefile = '~/cookies.txt'
    gdal_cookiejar = '~/cookies.txt'
    gdal_disable_read = 'EMPTY_DIR'
    gdal_curl_extensions = 'TIF'
    gdal_unsafessl = 'YES'
    
    gdal_config = {
    "GDAL_HTTP_COOKIEFILE": '~/cookies.txt',
    "GDAL_HTTP_COOKIEJAR": '~/cookies.txt',
    "GDAL_DISABLE_READDIR_ON_OPEN": 'YES',
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": 'TIF',
    "GDAL_HTTP_UNSAFESSL": 'YES',
    "CPL_VSIL_CURL_USE_HEAD":False,
    #"CPL_CURL_VERBOSE":True
    }
    max_retries = 1
    retry_delay = 5 # seconds to wait between retries
    for attempt in range(max_retries):
        try:
            with rasterio.Env(**gdal_config): # Setting up GDAL environment
                ds.load()
            
        except Exception as e:
            print(e)
    return ds


    

def load_hls_url(link,chunk_size):
    # create an empty dataset xarray
    gdal_config = {
    "GDAL_HTTP_COOKIEFILE": '~/cookies.txt',
    "GDAL_HTTP_COOKIEJAR": '~/cookies.txt',
    "GDAL_DISABLE_READDIR_ON_OPEN": 'YES',
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": 'TIF',
    "GDAL_HTTP_UNSAFESSL": 'YES',
    "CPL_VSIL_CURL_USE_HEAD":False,
    #"CPL_CURL_VERBOSE":True
    }
    
    rasterio.Env(**gdal_config)
    
    band = link.split('/')[-1].split('.')[-2]
    array = rxr.open_rasterio(link, chunks=chunk_size, masked=True, name = band).squeeze('band', drop=True).load()
    array.name = band
    if 'Fmask' not in band:
        array.attrs['scale_factor'] = 0.0001 
    else:
        array.attrs['scale_factor'] = 1
    
    return array
    
def get_geometry_clip(city_name):
    from shapely.geometry import box
    import geopandas as gpd
    
    df_geo = gpd.read_file('cb_2018_us_ua10_500k/cb_2018_us_ua10_500k.shp')
    df_geo_loc = df_geo.loc[df_geo['NAME10'] == city_name]
    
    # create a bounding box from the shapefile 
    bounds = df_geo_loc.geometry.bounds.values[0]
    geom = box(*bounds)
    
    df_geo_out = gpd.GeoDataFrame({"id":1,"geometry":[box(*bounds)]})
    df_geo_out = df_geo_out.set_geometry('geometry')
    df_geo_out.crs = df_geo.crs

    return df_geo_out

def scale_hls_data(ds,bands):

    for band in bands:

        if 'Fmask' not in band:
    
            ds[band].data = 0.0001 * ds[band].data

    return ds


def configure_gdal_rasterio_dask():
    # ----------------- Step 1: 1. Monkey‑patch Xarray’s .load() to wrap every read in your Env ----------------- 
    # 1. Grab the real load method *before* patching
    _orig_ds_load = xr.Dataset.load
    _orig_da_load = xr.DataArray.load
    _orig_da_compute = xr.Dataset.compute
    
    def _load_with_env(self, **kwargs):
        # 2. In your Env you can set any GDAL/Rasterio opts
        with Env(
            GDAL_DISABLE_READDIR_ON_OPEN = "EMPTY_DIR",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS = "TIF",
            GDAL_HTTP_COOKIEFILE = '~/cookies.txt',
            GDAL_HTTP_COOKIEJAR = '~/cookies.txt',
            GDAL_HTTP_UNSAFESSL = 'YES',
            GDAL_HTTP_MAX_RETRY = '10',
            GDAL_HTTP_RETRY_DELAY = '0.5',
            CPL_VSIL_CURL_USE_HEAD = 'YES'
        ):
            # 3. Call the *original* load, not xr.Dataset.load
            return _orig_ds_load(self, **kwargs)
    
    def _da_load_with_env(self, **kwargs):
        with Env(
            GDAL_DISABLE_READDIR_ON_OPEN = "EMPTY_DIR",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS = "TIF",
            GDAL_HTTP_COOKIEFILE = '~/cookies.txt',
            GDAL_HTTP_COOKIEJAR = '~/cookies.txt',
            GDAL_HTTP_UNSAFESSL = 'YES',
            GDAL_HTTP_MAX_RETRY = '10',
            GDAL_HTTP_RETRY_DELAY = '0.5',
            CPL_VSIL_CURL_USE_HEAD = 'YES'
        ):
            return _orig_da_load(self, **kwargs)
            
    def _da_compute_with_env(self, **kwargs):
        with Env(
            GDAL_DISABLE_READDIR_ON_OPEN = "EMPTY_DIR",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS = "TIF",
            GDAL_HTTP_COOKIEFILE = '~/cookies.txt',
            GDAL_HTTP_COOKIEJAR = '~/cookies.txt',
            GDAL_HTTP_UNSAFESSL = 'YES',
            GDAL_HTTP_MAX_RETRY = '10',
            GDAL_HTTP_RETRY_DELAY = '0.5',
            CPL_VSIL_CURL_USE_HEAD = 'YES'
        ):
            return _orig_da_compute(self, **kwargs)
    
    # 4. Now monkey‑patch
    xr.Dataset.load   = _load_with_env
    xr.DataArray.load = _da_load_with_env
    xr.Dataset.compute = _da_compute_with_env
    # ----------------- Step 2: Monkey‑patch rasterio.open itself ----------------- 
    

    # 1. Capture the true open
    _orig_open = rasterio.open
    
    def open_with_env(*args, **kwargs):
        with Env(
            GDAL_DISABLE_READDIR_ON_OPEN = "EMPTY_DIR",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS = "TIF",
            GDAL_HTTP_COOKIEFILE = '~/cookies.txt',
            GDAL_HTTP_COOKIEJAR = '~/cookies.txt',
            GDAL_HTTP_UNSAFESSL = 'YES',
            GDAL_HTTP_MAX_RETRY = '10',
            GDAL_HTTP_RETRY_DELAY = '0.5',
            CPL_VSIL_CURL_USE_HEAD = 'YES'
        ):
            # 2. Call real open
            return _orig_open(*args, **kwargs)
    
    # 3. Replace it
    rasterio.open = open_with_env

     # ----------------- Step 3: Ensure every Dask worker has the GDAL env set before they read ----------------- 
    
def _setup_gdal():
    
    os.environ.update({
        "GDAL_DISABLE_READDIR_ON_OPEN" : "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS" : "TIF",
        "GDAL_HTTP_COOKIEFILE" : '~/cookies.txt',
        "GDAL_HTTP_COOKIEJAR" : '~/cookies.txt',
        "GDAL_HTTP_UNSAFESSL" : 'YES',
        "GDAL_HTTP_MAX_RETRY" : '10',
        "GDAL_HTTP_RETRY_DELAY" : '0.5',
        "CPL_VSIL_CURL_USE_HEAD" : 'YES'
        })
    from rasterio.env import Env
    Env().__enter__()
    
    
def pystac_to_json(item_list):
    import pandas as pd
    from shapely.geometry import Polygon
    data = [i.geometry for i in item_list]
    parsed_data = [[item['type'],Polygon(item['coordinates'][0])] for item in data]
    df_geo = gpd.GeoDataFrame(data=parsed_data, columns=['type','geometry'])

    data_id = [i.id for i in item_list]
    df_geo['id'] = data_id
    df_geo['granule_id'] = [i.split('.')[2] for i in df_geo['id']]
    df_geo['sat_id'] = [i.split('.')[1] for i in df_geo['id']]
    df_geo['date'] = [i.split('.')[3].split('T')[0] for i in df_geo['id']]
    
    return df_geo


def load_data_into_memory(ds_mask_scaled,split_para=1000):
    max_attemp = int(len(ds_mask_scaled.x)/split_para)+2
    try:
        ds_mask_scaled_sel = ds_mask_scaled
        ds_mask_scaled_sel.load() 
    
    except:
        
        for i in range(max_attemp):
    
            if i==max_attemp:
                ds_mask_scaled_sel = ds_mask_scaled.sel(x=ds_mask_scaled.x[i:],
                                                    y=ds_mask_scaled.y[i:])
                
                ds_mask_scaled_sel = ds_mask_scaled_sel.chunk({'y':split_para*i/5,'x':split_para*i/5})
            else:
            
                ds_mask_scaled_sel = ds_mask_scaled.sel(x=ds_mask_scaled.x[i:split_para*i],
                                                    y=ds_mask_scaled.y[i:split_para*i])
    
                ds_mask_scaled_sel = ds_mask_scaled_sel.chunk({'y':split_para*i/5,'x':split_para*i/5})
            
            print(i,len(ds_mask_scaled_sel.x),len(ds_mask_scaled_sel.y))
            ds_mask_scaled_sel.load()

    return ds_mask_scaled_sel

    