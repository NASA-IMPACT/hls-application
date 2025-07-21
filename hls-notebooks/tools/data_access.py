'''
@author: Trang Vo - tv0015@uah.edu

Last modified: Mon Jul 21 11:39:04 2025

'''

import earthaccess
import pandas as pd
import numpy as np
import pystac_client
import odc.stac
import xarray as xr
import rasterio
from rasterio.env import Env
import os


def search_cmr_stac(start_year,end_year,lat_range,lon_range,date_begin=None,date_end=None,mgrs_tile_id=None):
    
    """
    Searches the CMR STAC API for HLS granules within a specified spatial and temporal range.

    This function queries the HLSL30.v2.0 and HLSS30.v2.0 collections from NASA's LPCLOUD STAC endpoint.
    The search is split by year to handle API limitations and is most suitable for small to medium spatial domains.

    Parameters:
    -----------
    start_year : int or str
        The beginning year of the search period. Can be an integer (e.g., 2020) or a string (e.g., '2020-01-01').
    end_year : int or str
        The ending year of the search period (inclusive). Same format as start_year.
    lat_range : list or tuple
        A pair of latitude values specifying the search bounds, e.g., [min_lat, max_lat].
    lon_range : list or tuple
        A pair of longitude values specifying the search bounds, e.g., [min_lon, max_lon].
    date_begin : str, optional
        Optional start date (MM-DD) for each year, e.g., '06-01'. If None, defaults to '01-01'.
    date_end : str, optional
        Optional end date (MM-DD) for each year, e.g., '09-30'. If None, defaults to '12-31'.
    mgrs_tile_id : str, optional
        If provided, filters the returned items by the MGRS tile ID.

    Returns:
    --------
    items_list : list
        A flat list of STAC item objects (granules) that match the search criteria.
    """
    
    bbox = ([min(lon_range),min(lat_range),max(lon_range),max(lat_range)])
    if type(start_year) is int:
        years = np.arange(start_year,end_year+1,1)
    else:
        years = np.arange(int(start_year.split('-')[0]),
                          int(end_year.split('-')[0])+1
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

        if mgrs_tile_id is not None:
            items = [item for item in items if mgrs_tile_id in item.id]
                    
        
        items_list.append(items)
        print('Found', len(items), 'granules at point', bbox, 'from', dt_min, 'to', dt_max)

    
    items_list = [i for item in items_list for i in item]

    return items_list


def rename_common_bands(items_list):
    """
    Renames the spectral band asset keys in a list of HLS STAC items to a common naming convention.

    This function standardizes band names for both Sentinel-2 (HLS.S30) and Landsat (HLS.L30) items by 
    mapping their band-specific keys to a unified set of descriptive names: 
    ['Blue', 'Green', 'Red', 'NIR', 'SWIR_1', 'SWIR_2'].

    For example:
    - HLS.S30 band 'B8A' → 'NIR'
    - HLS.L30 band 'B05' → 'NIR'

    Parameters:
    -----------
    items_list : list
        A list of STAC item objects containing HLS.L30 and/or HLS.S30 assets.

    Returns:
    --------
    items_list : list
        The same list of STAC items with band asset keys renamed to common names.
    """

    # Rename HLSS B8A and HLSL B05 to common band name
    S30_band_common = ['B02','B03','B04','B8A','B11','B12']
    L30_band_common = ['B02','B03','B04','B05','B06','B07']
    band_name = ['Blue','Green','Red','NIR','SWIR_1','SWIR_2']
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
    """
    Loads a spatiotemporal datacube from a list of STAC items using the ODC-STAC interface.

    This function sets the desired coordinate reference system (CRS), spatial resolution, and bounding box, 
    and loads the data lazily as an xarray Dataset with Dask chunking.

    Parameters:
    -----------
    crs : str or pyproj.CRS
        The target coordinate reference system for the output data (e.g., "EPSG:32614").
    
    bands : list of str
        List of band names (e.g., ['Blue', 'Green', 'Red']) to load from the STAC items.

    spatial_res : tuple of float
        The target spatial resolution as (x_resolution, y_resolution), e.g., (30, 30).

    items_list : list
        A list of STAC item objects (e.g., from pystac-client or search_cmr_stac) to be loaded.

    bbox : list or tuple
        Bounding box to constrain the spatial extent of the data, in the form [min_lon, min_lat, max_lon, max_lat].

    Returns:
    --------
    ds : xarray.Dataset
        A lazily-loaded xarray dataset containing the selected bands, spatially and temporally aligned.
    """

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
    

def get_geometry_clip(city_name):
    """
    Retrieves the bounding box geometry of a specified urban area (city) from a US Census shapefile.

    This function loads a shapefile of Urban Areas (2018 vintage from the US Census),
    filters it by the given city name, extracts its bounding box, and returns it as a GeoDataFrame.

    Parameters:
    -----------
    city_name : str
        The name of the city to extract the bounding box for, matching the 'NAME10' field in the shapefile.

    Returns:
    --------
    df_geo_out : geopandas.GeoDataFrame
        A GeoDataFrame containing a single rectangular geometry (bounding box) for the specified city.
        The CRS matches that of the original shapefile.

    Notes:
    ------
    - The shapefile must be located at `'cb_2018_us_ua10_500k/cb_2018_us_ua10_500k.shp'` relative to the script.
    - The returned geometry is a bounding box, not the full city polygon.
    """
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
    """
    Scales surface reflectance values in selected HLS bands to reflectance units.

    HLS data is provided as integer values scaled by a factor of 10,000. This function multiplies 
    the selected bands (excluding 'Fmask') by 0.0001 to convert them to reflectance values 
    in the range of approximately 0–1.

    Parameters:
    -----------
    ds : xarray.Dataset
        The dataset containing HLS bands as data variables.

    bands : list of str
        List of band names to scale (e.g., ['Blue', 'Green', 'Red', 'NIR']).
        Bands containing 'Fmask' in the name will be skipped.

    Returns:
    --------
    ds : xarray.Dataset
        The same dataset with the specified bands scaled to reflectance values.


    """

    for band in bands:

        if 'Fmask' not in band:
    
            ds[band].data = 0.0001 * ds[band].data

    return ds


def configure_gdal_rasterio_dask():
    """
    Configures GDAL, Rasterio, and Xarray to consistently use a custom GDAL environment 
    for reading remote datasets (e.g., cloud-hosted STAC assets).

    This function:
    1. Monkey-patches `xarray.Dataset.load()`, `xarray.DataArray.load()`, and 
       `xarray.Dataset.compute()` to ensure all data loading operations occur within 
       a custom GDAL environment using `rasterio.Env`.
    2. Monkey-patches `rasterio.open()` so any direct file access also respects this 
       GDAL configuration.

    The GDAL environment is customized with the following settings:
    - Enables reading cloud-optimized GeoTIFFs (COGs) over HTTP.
    - Uses a local cookie file (`~/cookies.txt`) for authenticated access.
    - Increases GDAL retry behavior for robustness against intermittent HTTP failures.
    - Disables directory reads on open to reduce HTTP overhead.

    These patches are helpful in workflows involving:
    - NASA's LP DAAC data accessed via STAC endpoints.
    - ODCE/Opendatacube-based STAC workflows using `odc.stac.stac_load()`.
    - Dask-enabled processing with remote assets and lazy loading.

    Dependencies:
    ------------
    - `xarray`
    - `rasterio`
    - `rasterio.Env` from `rasterio.env`
    """

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

def load_data_into_memory(ds_mask_scaled,split_para=1000):
    
    """
    Attempts to load a large xarray.Dataset into memory in chunks to avoid memory or I/O-related failures.

    If a full dataset `.load()` fails (e.g., due to size or remote data access limits), this function 
    falls back to iteratively loading spatial chunks along the x and y dimensions.

    Parameters:
    -----------
    ds_mask_scaled : xarray.Dataset
        A dataset with spatial dimensions ('x', 'y') that is lazily loaded (e.g., from a remote source or Dask).
    
    split_para : int, optional (default=1000)
        Defines the spatial chunk size used during fallback loading. Larger values load more data per attempt.

    Returns:
    --------
    ds_mask_scaled_sel : xarray.Dataset
        A dataset that has been loaded into memory, either in full or via spatial chunks.


    Example:
    --------
    >>> ds = odc.stac.stac_load(...)
    >>> ds_scaled = scale_hls_data(ds, bands)
    >>> ds_loaded = load_data_into_memory(ds_scaled, split_para=512)
    """
    
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