#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  7 09:00:50 2025

@author: tvo
"""

# Import required packages
import math
import folium
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib import colors as mcolours
from matplotlib.animation import FuncAnimation
from pathlib import Path
from pyproj import Transformer
from shapely.geometry import box
from skimage.exposure import rescale_intensity



def _degree_to_zoom_level(l1, l2, margin=0.0):
    """
    Helper function to set zoom level for `display_map`
    """

    degree = abs(l1 - l2) * (1 + margin)
    zoom_level_int = 0
    if degree != 0:
        zoom_level_float = math.log(360 / degree) / math.log(2)
        zoom_level_int = int(zoom_level_float)
    else:
        zoom_level_int = 18
    return zoom_level_int

def display_map(x, y, crs='EPSG:4326', margin=-0.5, zoom_bias=0, centroid=None):
    """ 
    Given a set of x and y coordinates, this function generates an 
    interactive map with a bounded rectangle overlayed on Google Maps 
    imagery.        
    
    Last modified: February 2023
    
    Modified from function written by Otto Wagner available here: 
    https://github.com/ceos-seo/data_cube_utilities/tree/master/data_cube_utilities
    
    Parameters
    ----------  
    x : (float, float)
        A tuple of x coordinates in (min, max) format. 
    y : (float, float)
        A tuple of y coordinates in (min, max) format.
    crs : string, optional
        A string giving the EPSG CRS code of the supplied coordinates. 
        The default is 'EPSG:4326'.
    margin : float
        A numeric value giving the number of degrees lat-long to pad 
        the edges of the rectangular overlay polygon. A larger value 
        results more space between the edge of the plot and the sides 
        of the polygon. Defaults to -0.5.
    zoom_bias : float or int
        A numeric value allowing you to increase or decrease the zoom 
        level by one step. Defaults to 0; set to greater than 0 to zoom 
        in, and less than 0 to zoom out.
        
    Returns
    -------
    folium.Map : A map centered on the supplied coordinate bounds. A 
    rectangle is drawn on this map detailing the perimeter of the x, y 
    bounds.  A zoom level is calculated such that the resulting 
    viewport is the closest it can possibly get to the centered 
    bounding rectangle without clipping it. 
    """

    # Convert each corner coordinates to lat-lon
    all_x = (x[0], x[1], x[0], x[1])
    all_y = (y[0], y[0], y[1], y[1])
    transformer = Transformer.from_crs(crs, "EPSG:4326")
    all_longitude, all_latitude = transformer.transform(all_x, all_y)

    # Calculate zoom level based on coordinates
    lat_zoom_level = _degree_to_zoom_level(
        min(all_latitude), max(all_latitude), margin=margin) + zoom_bias
    lon_zoom_level = _degree_to_zoom_level(
        min(all_longitude), max(all_longitude), margin=margin) + zoom_bias
    zoom_level = min(lat_zoom_level, lon_zoom_level)

    # Identify centre point for plotting
    center = [np.mean(all_latitude), np.mean(all_longitude)]

    # Create map
    interactive_map = folium.Map(
        location=center,
        zoom_start=zoom_level,
        tiles="http://mt1.google.com/vt/lyrs=y&z={z}&x={x}&y={y}",
        attr="Google")

    # Create bounding box coordinates to overlay on map
    line_segments = [(all_latitude[0], all_longitude[0]),
                     (all_latitude[1], all_longitude[1]),
                     (all_latitude[3], all_longitude[3]),
                     (all_latitude[2], all_longitude[2]),
                     (all_latitude[0], all_longitude[0])]

    

    # Add the centroid point as an overlay 
    if centroid is not None:
        interactive_map.add_child(
        folium.Circle(location=[centroid[0],centroid[-1]],
                                 color='red',
                                 opacity=1,
                                radius=10000,
                               fill=False
                              ),
        
        
        )



        
    else:
        # Add bounding box as an overlay
        interactive_map.add_child(
        folium.features.PolyLine(locations=line_segments,
                                 color='red',
                                 opacity=0.8))
        

    # Add clickable lat-lon popup box
    interactive_map.add_child(folium.features.LatLngPopup())

    return interactive_map

def rgb(ds,
        bands=['nbart_red', 'nbart_green', 'nbart_blue'],
        index=None,
        index_dim='time',
        robust=True,
        percentile_stretch=None,
        col_wrap=4,
        size=6,
        aspect=None,
        titles=None,
        savefig_path=None,
        savefig_kwargs={},
        **kwargs):
    """
    Takes an xarray dataset and plots RGB images using three imagery 
    bands (e.g ['nbart_red', 'nbart_green', 'nbart_blue']). The `index` 
    parameter allows easily selecting individual or multiple images for 
    RGB plotting. Images can be saved to file by specifying an output 
    path using `savefig_path`.
    
    This function was designed to work as an easier-to-use wrapper 
    around xarray's `.plot.imshow()` functionality.
    
    Last modified: September 2020
    
    Parameters
    ----------  
    ds : xarray Dataset
        A two-dimensional or multi-dimensional array to plot as an RGB 
        image. If the array has more than two dimensions (e.g. multiple 
        observations along a 'time' dimension), either use `index` to 
        select one (`index=0`) or multiple observations 
        (`index=[0, 1]`), or create a custom faceted plot using e.g. 
        `col="time"`.       
    bands : list of strings, optional
        A list of three strings giving the band names to plot. Defaults 
        to '['nbart_red', 'nbart_green', 'nbart_blue']'.
    index : integer or list of integers, optional
        `index` can be used to select one (`index=0`) or multiple 
        observations (`index=[0, 1]`) from the input dataset for 
        plotting. If multiple images are requested these will be plotted
        as a faceted plot.
    index_dim : string, optional
        The dimension along which observations should be plotted if 
        multiple observations are requested using `index`. Defaults to 
        `time`.
    robust : bool, optional
        Produces an enhanced image where the colormap range is computed 
        with 2nd and 98th percentiles instead of the extreme values. 
        Defaults to True.
    percentile_stretch : tuple of floats
        An tuple of two floats (between 0.00 and 1.00) that can be used 
        to clip the colormap range to manually specified percentiles to 
        get more control over the brightness and contrast of the image. 
        The default is None; '(0.02, 0.98)' is equivelent to 
        `robust=True`. If this parameter is used, `robust` will have no 
        effect.
    col_wrap : integer, optional
        The number of columns allowed in faceted plots. Defaults to 4.
    size : integer, optional
        The height (in inches) of each plot. Defaults to 6.
    aspect : integer, optional
        Aspect ratio of each facet in the plot, so that aspect * size 
        gives width of each facet in inches. Defaults to None, which 
        will calculate the aspect based on the x and y dimensions of 
        the input data.
    titles : string or list of strings, optional
        Replace the xarray 'time' dimension on plot titles with a string
        or list of string titles, when a list of index values are
        provided, of your choice. Defaults to None.
    savefig_path : string, optional
        Path to export image file for the RGB plot. Defaults to None, 
        which does not export an image file.
    savefig_kwargs : dict, optional
        A dict of keyword arguments to pass to 
        `matplotlib.pyplot.savefig` when exporting an image file. For 
        all available options, see: 
        https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html        
    **kwargs : optional
        Additional keyword arguments to pass to `xarray.plot.imshow()`.
        For example, the function can be used to plot into an existing
        matplotlib axes object by passing an `ax` keyword argument.
        For more options, see:
        http://xarray.pydata.org/en/stable/generated/xarray.plot.imshow.html  
        
    Returns
    -------
    An RGB plot of one or multiple observations, and optionally an image
    file written to file.
    
    """
    
    # Get names of x and y dims
    y_dim, x_dim = ds.odc.spatial_dims

    # If ax is supplied via kwargs, ignore aspect and size
    if 'ax' in kwargs:

        # Create empty aspect size kwarg that will be passed to imshow
        aspect_size_kwarg = {}
    else:
        # Compute image aspect
        if not aspect:
            aspect = ds.odc.geobox.aspect

        # Populate aspect size kwarg with aspect and size data
        aspect_size_kwarg = {'aspect': aspect, 'size': size}

    # If no value is supplied for `index` (the default), plot using default
    # values and arguments passed via `**kwargs`
    if index is None:

        # Select bands and convert to DataArray
        da = ds[bands].to_array().compute()

        # If percentile_stretch == True, clip plotting to percentile vmin, vmax
        if percentile_stretch:
            vmin, vmax = da.quantile(percentile_stretch).values
            kwargs.update({'vmin': vmin, 'vmax': vmax})

        # If there are more than three dimensions and the index dimension == 1,
        # squeeze this dimension out to remove it
        if ((len(ds.dims) > 2) and ('col' not in kwargs) and
            (len(da[index_dim]) == 1)):

            da = da.squeeze(dim=index_dim)

        # If there are more than three dimensions and the index dimension
        # is longer than 1, raise exception to tell user to use 'col'/`index`
        elif ((len(ds.dims) > 2) and ('col' not in kwargs) and
              (len(da[index_dim]) > 1)):

            raise Exception(
                f'The input dataset `ds` has more than two dimensions: '
                f'{list(ds.dims.keys())}. Please select a single observation '
                'using e.g. `index=0`, or enable faceted plotting by adding '
                'the arguments e.g. `col="time", col_wrap=4` to the function '
                'call')

        img = da.plot.imshow(x=x_dim,
                             y=y_dim,
                             robust=robust,
                             col_wrap=col_wrap,
                             **aspect_size_kwarg,
                             **kwargs)
        if titles is not None:
            for ax, title in zip(img.axs.flat, titles):
                ax.set_title(title)

    # If values provided for `index`, extract corresponding observations and
    # plot as either single image or facet plot
    else:

        # If a float is supplied instead of an integer index, raise exception
        if isinstance(index, float):
            raise Exception(
                f'Please supply `index` as either an integer or a list of '
                'integers')

        # If col argument is supplied as well as `index`, raise exception
        if 'col' in kwargs:
            raise Exception(
                f'Cannot supply both `index` and `col`; please remove one and '
                'try again')

        # Convert index to generic type list so that number of indices supplied
        # can be computed
        index = index if isinstance(index, list) else [index]

        # Select bands and observations and convert to DataArray
        da = ds[bands].isel(**{index_dim: index}).to_array().compute()

        # If percentile_stretch == True, clip plotting to percentile vmin, vmax
        if percentile_stretch:
            vmin, vmax = da.quantile(percentile_stretch).values
            kwargs.update({'vmin': vmin, 'vmax': vmax})

        # If multiple index values are supplied, plot as a faceted plot
        if len(index) > 1:

            img = da.plot.imshow(x=x_dim,
                                 y=y_dim,
                                 robust=robust,
                                 col=index_dim,
                                 col_wrap=col_wrap,
                                 **aspect_size_kwarg,
                                 **kwargs)
            if titles is not None:
                for ax, title in zip(img.axs.flat, titles):
                    ax.set_title(title)

        # If only one index is supplied, squeeze out index_dim and plot as a
        # single panel
        else:

            img = da.squeeze(dim=index_dim).plot.imshow(robust=robust,
                                                        **aspect_size_kwarg,
                                                        **kwargs)
            if titles is not None:
                for ax, title in zip(img.axs.flat, titles):
                    ax.set_title(title)

    # If an export path is provided, save image to file. Individual and
    # faceted plots have a different API (figure vs fig) so we get around this
    # using a try statement:
    if savefig_path:

        print(f'Exporting image to {savefig_path}')

        try:
            img.fig.savefig(savefig_path, **savefig_kwargs)
        except:
            img.figure.savefig(savefig_path, **savefig_kwargs)

def single_band(ds,
        band=None,
        index=None,
        index_dim='time',
        robust=True,
        vmin=None,
        vmax=None,
        label=None,
        col_wrap=4,
        size=6,
        aspect=None,
        titles=None,
        savefig_path=None,
        savefig_kwargs={},
        **kwargs):
    """
    
    Parameters
    ----------  
    ds : xarray datarray
        A two-dimensional or multi-dimensional array to plot as an RGB 
        image. If the array has more than two dimensions (e.g. multiple 
        observations along a 'time' dimension), either use `index` to 
        select one (`index=0`) or multiple observations 
        (`index=[0, 1]`), or create a custom faceted plot using e.g. 
        `col="time"`.       
    bands : list of strings, optional
        A list of three strings giving the band names to plot. Defaults 
        to '['nbart_red', 'nbart_green', 'nbart_blue']'.
    index : integer or list of integers, optional
        `index` can be used to select one (`index=0`) or multiple 
        observations (`index=[0, 1]`) from the input dataset for 
        plotting. If multiple images are requested these will be plotted
        as a faceted plot.
    index_dim : string, optional
        The dimension along which observations should be plotted if 
        multiple observations are requested using `index`. Defaults to 
        `time`.
    robust : bool, optional
        Produces an enhanced image where the colormap range is computed 
        with 2nd and 98th percentiles instead of the extreme values. 
        Defaults to True.
    percentile_stretch : tuple of floats
        An tuple of two floats (between 0.00 and 1.00) that can be used 
        to clip the colormap range to manually specified percentiles to 
        get more control over the brightness and contrast of the image. 
        The default is None; '(0.02, 0.98)' is equivelent to 
        `robust=True`. If this parameter is used, `robust` will have no 
        effect.
    col_wrap : integer, optional
        The number of columns allowed in faceted plots. Defaults to 4.
    size : integer, optional
        The height (in inches) of each plot. Defaults to 6.
    aspect : integer, optional
        Aspect ratio of each facet in the plot, so that aspect * size 
        gives width of each facet in inches. Defaults to None, which 
        will calculate the aspect based on the x and y dimensions of 
        the input data.
    titles : string or list of strings, optional
        Replace the xarray 'time' dimension on plot titles with a string
        or list of string titles, when a list of index values are
        provided, of your choice. Defaults to None.
    savefig_path : string, optional
        Path to export image file for the RGB plot. Defaults to None, 
        which does not export an image file.
    savefig_kwargs : dict, optional
        A dict of keyword arguments to pass to 
        `matplotlib.pyplot.savefig` when exporting an image file. For 
        all available options, see: 
        https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html        
    **kwargs : optional
        Additional keyword arguments to pass to `xarray.plot.imshow()`.
        For example, the function can be used to plot into an existing
        matplotlib axes object by passing an `ax` keyword argument.
        For more options, see:
        http://xarray.pydata.org/en/stable/generated/xarray.plot.imshow.html  
        
    Returns
    -------
    An RGB plot of one or multiple observations, and optionally an image
    file written to file.
    
    """
    
    # Get names of x and y dims
    y_dim, x_dim = ds.odc.spatial_dims

    # If ax is supplied via kwargs, ignore aspect and size
    if 'ax' in kwargs:

        # Create empty aspect size kwarg that will be passed to imshow
        aspect_size_kwarg = {}
    else:
        # Compute image aspect
        if not aspect:
            aspect = ds.odc.geobox.aspect

        # Populate aspect size kwarg with aspect and size data
        aspect_size_kwarg = {'aspect': aspect, 'size': size}

    # If no value is supplied for `index` (the default), plot using default
    # values and arguments passed via `**kwargs`
    if index is None:

        # Select bands and convert to DataArray
        da = ds.to_array().compute()

        # If percentile_stretch == True, clip plotting to percentile vmin, vmax
        
        kwargs.update({'vmin': vmin, 'vmax': vmax})

        # If there are more than three dimensions and the index dimension == 1,
        # squeeze this dimension out to remove it
        if ((len(ds.dims) > 2) and ('col' not in kwargs) and
            (len(da[index_dim]) == 1)):

            da = da.squeeze(dim=index_dim)

        # If there are more than three dimensions and the index dimension
        # is longer than 1, raise exception to tell user to use 'col'/`index`
        elif ((len(ds.dims) > 2) and ('col' not in kwargs) and
              (len(da[index_dim]) > 1)):

            raise Exception(
                f'The input dataset `ds` has more than two dimensions: '
                f'{list(ds.dims.keys())}. Please select a single observation '
                'using e.g. `index=0`, or enable faceted plotting by adding '
                'the arguments e.g. `col="time", col_wrap=4` to the function '
                'call')

        img = da.plot.imshow(x=x_dim,
                             y=y_dim,
                             robust=robust,
                             col_wrap=col_wrap,
                             **aspect_size_kwarg,
                             **kwargs)
        if titles is not None:
            for ax, title in zip(img.axs.flat, titles):
                ax.set_title(title,fontsize=22)
        img.cbar.ax.tick_params(labelsize=30)
        img.cbar.set_label(label=label, size=30, weight='bold')

    # If values provided for `index`, extract corresponding observations and
    # plot as either single image or facet plot
    else:

        # If a float is supplied instead of an integer index, raise exception
        if isinstance(index, float):
            raise Exception(
                f'Please supply `index` as either an integer or a list of '
                'integers')

        # If col argument is supplied as well as `index`, raise exception
        if 'col' in kwargs:
            raise Exception(
                f'Cannot supply both `index` and `col`; please remove one and '
                'try again')

        # Convert index to generic type list so that number of indices supplied
        # can be computed
        index = index if isinstance(index, list) else [index]

        # Select bands and observations and convert to DataArray
        da = ds.isel(**{index_dim: index}).compute()

        # If percentile_stretch == True, clip plotting to percentile vmin, vmax

        kwargs.update({'vmin': vmin, 'vmax': vmax})

        # If multiple index values are supplied, plot as a faceted plot
        if len(index) > 1:

            img = da.plot.imshow(x=x_dim,
                                 y=y_dim,
                                 robust=robust,
                                 col=index_dim,
                                 col_wrap=col_wrap,
                                 **aspect_size_kwarg,
                                 **kwargs)
            if titles is not None:
                for ax, title in zip(img.axs.flat, titles):
                    ax.set_title(title,fontsize=22)

            img.cbar.ax.tick_params(labelsize=30)
            img.cbar.set_label(label=label, size=30, weight='bold')

        # If only one index is supplied, squeeze out index_dim and plot as a
        # single panel
        else:

            img = da.squeeze(dim=index_dim).plot.imshow(robust=robust,
                                                        **aspect_size_kwarg,
                                                        **kwargs)
            if titles is not None:
                for ax, title in zip(img.axs.flat, titles):
                    ax.set_title(title,fontsize=22)
    

            img.cbar.ax.tick_params(labelsize=30)
            img.cbar.set_label(label=label, size=30, weight='bold')
    # If an export path is provided, save image to file. Individual and
    # faceted plots have a different API (figure vs fig) so we get around this
    # using a try statement:
    if savefig_path:

        print(f'Exporting image to {savefig_path}')

        try:
            img.fig.savefig(savefig_path, **savefig_kwargs)
        except:
            img.figure.savefig(savefig_path, **savefig_kwargs)
    return img


def urban_growth_plot(ds,urban_area,baseline_year,analysis_year):
     
    # Plot urban extent from first year in grey as a background
    plot = ds.ENDISI.sel(year=baseline_year).plot(cmap='Greys',
                                           size=6,
                                           aspect=ds.y.size / ds.y.size,
                                           add_colorbar=False,
                                          
                                          )
  
    # Plot the meaningful change in urban area
    to_urban = '#b91e1e'
    urban_area_diff = urban_area.sel(year=analysis_year)-urban_area.sel(year=baseline_year)
    xr.where(urban_area_diff == 1, 1,
             np.nan).plot(ax=plot.axes,
                          add_colorbar=False,
                          cmap=ListedColormap([to_urban]))
    
    # Add the legend
    plot.axes.legend([Patch(facecolor=to_urban),
                      Patch(facecolor='darkgrey'),
                      Patch(facecolor='white')],
                     ['Urban growth hotspots', 'Remains urban'])
    plt.title('Urban growth between ' + str(baseline_year) + ' and ' +
              str(analysis_year));
    