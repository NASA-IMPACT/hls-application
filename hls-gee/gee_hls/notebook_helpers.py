"""Reusable helpers for the HLS + CCDC forest monitoring notebook."""

from __future__ import annotations

import struct
from collections import Counter, deque
from datetime import datetime, timedelta

import ee
import geemap
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


S2_BANDS_IN = ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12']
L8_BANDS_IN = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
COMMON_BANDS = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']
SEG_COLORS = ['#E53935', '#1E88E5', '#43A047', '#FB8C00',
              '#8E24AA', '#00ACC1', '#6D4C41', '#546E7A']
MOVING_AVG_WINDOW_DAYS = 120
MOVING_AVG_MIN_PERIODS = 14


def read_polygon_shapefile(shp_path):
    """Read polygon rings directly from a shapefile geometry file."""
    polygons = []
    with open(shp_path, 'rb') as f:
        header = f.read(100)
        if len(header) < 100:
            raise ValueError('Invalid shapefile header.')

        while True:
            record_header = f.read(8)
            if not record_header:
                break
            if len(record_header) < 8:
                raise ValueError('Corrupt shapefile record header.')

            _, content_len_words = struct.unpack('>2i', record_header)
            content = f.read(content_len_words * 2)
            if len(content) < 4:
                continue

            shape_type = struct.unpack('<i', content[:4])[0]
            if shape_type == 0:
                continue
            if shape_type not in (5, 15, 25):
                continue

            num_parts = struct.unpack('<i', content[36:40])[0]
            num_points = struct.unpack('<i', content[40:44])[0]
            parts = struct.unpack(f'<{num_parts}i', content[44:44 + 4 * num_parts])

            points_offset = 44 + 4 * num_parts
            points = np.frombuffer(content, dtype='<f8', offset=points_offset, count=num_points * 2)
            points = points.reshape((num_points, 2))

            for part_idx, start in enumerate(parts):
                end = parts[part_idx + 1] if part_idx + 1 < num_parts else num_points
                ring = points[start:end]
                if len(ring) >= 3:
                    polygons.append(ring.copy())

    return polygons


def count_clear_obs(point, start, end):
    """Count cloud-free L30, S30, and combined HLS images at a point."""
    l30_all = (ee.ImageCollection('NASA/HLS/HLSL30/v002')
               .filterBounds(point)
               .filterDate(start, end))
    s30_all = (ee.ImageCollection('NASA/HLS/HLSS30/v002')
               .filterBounds(point)
               .filterDate(start, end))

    def is_clear(img):
        fmask = img.select('Fmask')
        return img.set(
            'clear',
            fmask.eq(64).Or(fmask.eq(128))
            .reduceRegion(ee.Reducer.first(), point, 30)
            .get('Fmask')
        )

    l30_tagged = l30_all.map(is_clear)
    s30_tagged = s30_all.map(is_clear)

    n_l30_total = l30_all.size().getInfo()
    n_s30_total = s30_all.size().getInfo()
    n_l30_clear = l30_tagged.filter(ee.Filter.eq('clear', 1)).size().getInfo()
    n_s30_clear = s30_tagged.filter(ee.Filter.eq('clear', 1)).size().getInfo()

    return {
        'l30_total': n_l30_total,
        'l30_clear': n_l30_clear,
        's30_total': n_s30_total,
        's30_clear': n_s30_clear,
        'hls_total': n_l30_total + n_s30_total,
        'hls_clear': n_l30_clear + n_s30_clear,
    }


def mask_hls(image):
    """Keep clear-sky pixels using HLS Fmask."""
    fmask = image.select('Fmask')
    clear = fmask.eq(64).Or(fmask.eq(128))
    return image.updateMask(clear)


def prepare_s30(image):
    return (mask_hls(image)
            .select(S2_BANDS_IN, COMMON_BANDS)
            .multiply(0.0001)
            .copyProperties(image, ['system:time_start']))


def prepare_l30(image):
    return (mask_hls(image)
            .select(L8_BANDS_IN, COMMON_BANDS)
            .multiply(0.0001)
            .copyProperties(image, ['system:time_start']))


def add_indices(image):
    """Add NDVI, NBR, and EVI bands."""
    ndvi = image.normalizedDifference(['NIR', 'RED']).rename('NDVI')
    nbr = image.normalizedDifference(['NIR', 'SWIR2']).rename('NBR')
    evi = image.expression(
        '2.5 * (NIR - RED) / (NIR + 6.0*RED - 7.5*BLUE + 1.0)',
        {'NIR': image.select('NIR'), 'RED': image.select('RED'), 'BLUE': image.select('BLUE')}
    ).rename('EVI')
    return image.addBands([ndvi, nbr, evi])


def load_hls_collections(config):
    """Load L30, S30, and merged HLS collections for a site config."""
    start = config['start_date']
    end = config['end_date']
    region = config['region']

    s30 = (ee.ImageCollection('NASA/HLS/HLSS30/v002')
           .filterDate(start, end)
           .filterBounds(region)
           .map(prepare_s30)
           .map(add_indices))

    l30 = (ee.ImageCollection('NASA/HLS/HLSL30/v002')
           .filterDate(start, end)
           .filterBounds(region)
           .map(prepare_l30)
           .map(add_indices))

    hls = s30.merge(l30).sort('system:time_start')
    return l30, s30, hls


def get_dates(collection):
    """Return observation datetimes for an EE image collection."""
    ms_list = collection.aggregate_array('system:time_start').getInfo()
    return [datetime.utcfromtimestamp(ms / 1000) for ms in ms_list]


def count_by_year(dates, start_year, end_year):
    """Count unique observation days by year."""
    unique_days = {d.strftime('%Y-%m-%d') for d in dates}
    year_counts = Counter(d[:4] for d in unique_days)
    years = list(range(start_year, end_year + 1))
    counts = [year_counts.get(str(yr), 0) for yr in years]
    return years, counts


def to_decimal_year(dt):
    """Convert datetime to decimal year."""
    year = dt.year
    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    return year + (dt - start).total_seconds() / (end - start).total_seconds()


def decimal_to_datetime(dy):
    """Convert decimal year to datetime."""
    year = int(dy)
    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    delta = timedelta(seconds=(dy - year) * (end - start).total_seconds())
    return start + delta


def harmonic_model(t, coefs):
    """Evaluate the 3-harmonic CCDC model at decimal-year time values."""
    t = np.asarray(t, dtype=float)
    omega = 2.0 * np.pi
    return (
        coefs[0]
        + coefs[1] * t
        + coefs[2] * np.cos(t * omega)
        + coefs[3] * np.sin(t * omega)
        + coefs[4] * np.cos(t * omega * 2)
        + coefs[5] * np.sin(t * omega * 2)
        + coefs[6] * np.cos(t * omega * 3)
        + coefs[7] * np.sin(t * omega * 3)
    )


def get_pixel_ts(collection, point, bands):
    """Extract a point time series using getRegion."""
    raw = collection.select(bands).getRegion(point, 30).getInfo()
    if not raw or len(raw) < 2:
        return pd.DataFrame(columns=['date'] + list(bands))

    headers = raw[0]
    df = pd.DataFrame(raw[1:], columns=headers)
    df['date'] = pd.to_datetime(df['time'], unit='ms', utc=True).dt.tz_localize(None)
    df = df.dropna(subset=bands)
    df = df.sort_values('date').reset_index(drop=True)
    return df[['date'] + list(bands)]


def run_ccdc_at_point(collection, point, config):
    """Run CCDC over a tiny buffered point region and return the reduced result dict."""
    region = point.buffer(45).bounds()
    ccd = ee.Algorithms.TemporalSegmentation.Ccdc(
        collection=collection.filterBounds(region),
        breakpointBands=config['breakpoint_bands'],
        tmaskBands=['GREEN', 'SWIR2'],
        minObservations=config['min_observations'],
        dateFormat=1,
    )
    return ccd.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=point,
        scale=30,
        maxPixels=1e6,
    ).getInfo()


def get_ccdc_coefs(ccdc, band, seg_idx):
    """Extract eight CCDC coefficients for a segment from nested or flat output."""
    coefs_all = (ccdc or {}).get(f'{band}_coefs') or []
    if not coefs_all:
        return None
    first = coefs_all[0]
    if isinstance(first, (list, tuple)):
        if seg_idx < len(coefs_all) and len(coefs_all[seg_idx]) >= 8:
            return list(coefs_all[seg_idx])[:8]
    elif isinstance(first, (int, float)):
        start = seg_idx * 8
        chunk = coefs_all[start:start + 8]
        if len(chunk) == 8:
            return chunk
    return None


def add_smooth_fit(ax, ts_df, ccdc, band, seg_colors, smooth_factor=0.4):
    """Add a per-segment smoothing spline through observed points."""
    from scipy.interpolate import UnivariateSpline

    if ts_df is None or ts_df.empty or band not in ts_df.columns:
        return

    t_starts = (ccdc or {}).get('tStart', []) or []
    t_ends = (ccdc or {}).get('tEnd', []) or []
    t_breaks = (ccdc or {}).get('tBreak', []) or []

    df = ts_df[['date', band]].dropna().copy()
    df['t'] = df['date'].apply(to_decimal_year)
    df = df.sort_values('t').reset_index(drop=True)

    for i, (ts_i, te_i) in enumerate(zip(t_starts, t_ends)):
        seg = df[(df['t'] >= ts_i) & (df['t'] <= te_i)]
        if len(seg) < 3:
            continue

        x = seg['t'].values
        y = seg[band].values
        color = seg_colors[i % len(seg_colors)]
        x_dense = np.linspace(x[0], x[-1], 400)

        try:
            s = len(x) * np.var(y) * smooth_factor
            spline = UnivariateSpline(x, y, k=min(3, len(x) - 1), s=max(s, 1e-9))
            y_smooth = spline(x_dense)
        except Exception:
            deg = min(2, len(x) - 1)
            y_smooth = np.polyval(np.polyfit(x, y, deg), x_dense)

        dt_dense = [decimal_to_datetime(t) for t in x_dense]
        ax.plot(dt_dense, y_smooth, '-', color=color, lw=2.5,
                label=f'Seg {i + 1}' if i < 4 else '_nolegend_', zorder=3)

    for tb in t_breaks:
        if tb and tb > 0:
            ax.axvline(decimal_to_datetime(tb), color='red', lw=1.5, linestyle='--', alpha=0.85, zorder=4)


def plot_band_figure(datasets, band, config):
    """Create a multi-row pixel time-series figure for one band."""
    n_rows = len(datasets)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4.2 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for row, (label, ccdc, ts, color) in enumerate(datasets):
        ax = axes[row]
        if not ts.empty and band in ts.columns:
            ax.scatter(ts['date'], ts[band], c=color, s=12, alpha=0.55, zorder=2, label='Obs')
        add_smooth_fit(ax, ts, ccdc, band, SEG_COLORS)
        n_obs = len(ts) if not ts.empty else 0
        ax.set_ylabel(f'{label}\n({n_obs} obs)', fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='lower right', fontsize=7, framealpha=0.6)
        ax.margins(y=0.08)

        if row == n_rows - 1:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    bp_patch = plt.Line2D([0], [0], color='red', lw=1.5, linestyle='--', label='CCDC breakpoint')
    fig.legend(handles=[bp_patch], loc='upper right', fontsize=10, framealpha=0.8)
    fig.suptitle(
        f'{config["site_name"]}  |  lon={config["pixel_lon"]}, lat={config["pixel_lat"]}\n'
        f'Pixel-level CCDC — {config["start_year"]}–{config["end_year"]}  [{band}]',
        fontsize=13, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    return fig


def build_moving_average_fit(ts_df, band, start_date, end_date,
                             window_days=MOVING_AVG_WINDOW_DAYS,
                             min_periods=MOVING_AVG_MIN_PERIODS):
    """Build a centered moving-average fit from the HLS pixel time series."""
    df = ts_df[['date', band]].dropna().copy()
    if df.empty:
        return [], np.array([])
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.sort_values('date').groupby('date', as_index=False)[band].mean()

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    df = df[(df['date'] >= start_ts) & (df['date'] <= end_ts)]
    if df.empty:
        return [], np.array([])

    daily_index = pd.date_range(start_ts, end_ts, freq='D')
    daily = (df.set_index('date')[band]
               .reindex(daily_index)
               .interpolate(method='time', limit_area='inside'))
    smoothed = daily.rolling(f'{window_days}D', center=True, min_periods=min_periods).mean()
    valid = smoothed.dropna()
    return list(valid.index.to_pydatetime()), valid.to_numpy()


def load_or_compute_ccdc(collection, region, config, asset_path=None):
    """Load a precomputed CCDC asset or compute CCDC on the fly."""
    if asset_path:
        print(f'  Loading asset: {asset_path}')
        return ee.Image(asset_path).clip(region)

    print('  Computing on-the-fly (may be slow for large regions)...')
    ccd = ee.Algorithms.TemporalSegmentation.Ccdc(
        collection=collection.filterBounds(region),
        breakpointBands=config['breakpoint_bands'],
        tmaskBands=['GREEN', 'SWIR2'],
        minObservations=config['min_observations'],
        dateFormat=1,
    )
    return ccd.clip(region)


def ccdc_array_to_scalar(ccd_image, band):
    """Reduce CCDC array bands to scalar year/magnitude/count bands."""
    mag_arr = ccd_image.select(f'{band}_magnitude')
    tbreak_arr = ccd_image.select('tBreak')

    num_breaks = (tbreak_arr.gt(0)
                  .arrayReduce(ee.Reducer.sum(), [0])
                  .arrayGet([0])
                  .rename('num_breaks'))

    sentinel_0 = ee.Image(ee.Array([0.0]))
    mag_padded = mag_arr.arrayCat(sentinel_0, 0)
    tbreak_padded = tbreak_arr.arrayCat(sentinel_0, 0)

    abs_mag_p = mag_padded.abs()
    tbreak_sorted = tbreak_padded.arraySort(abs_mag_p.multiply(-1))
    mag_sorted = abs_mag_p.arraySort(abs_mag_p.multiply(-1))

    best_tbreak = tbreak_sorted.arrayGet([0])
    best_mag = mag_sorted.arrayGet([0])
    is_real_break = best_tbreak.gt(0)

    year_img = best_tbreak.floor().updateMask(is_real_break).rename('year_of_max_change')
    mag_img = best_mag.updateMask(is_real_break).rename('max_magnitude')
    return year_img.addBands(mag_img).addBands(num_breaks)


def scalar_ccdc_to_gdf(scalar_img, region, scale=150, tiles=1):
    """Sample a scalar CCDC image to a GeoDataFrame."""
    fc = scalar_img.sample(region=region, scale=scale, geometries=True)

    def pixel_to_box(feature):
        return feature.setGeometry(feature.geometry().buffer(scale / 2).bounds())

    fc = fc.map(pixel_to_box)
    df = geemap.ee_to_gdf(fc)
    if df.empty:
        print('  Warning: no pixels returned. Try increasing scale or checking region.')
    return df


def clean_disturbance_df(df):
    """Return a copy with numeric year values and valid geometries only."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if 'year_of_max_change' in out.columns:
        out['year_of_max_change'] = pd.to_numeric(out['year_of_max_change'], errors='coerce')
    if 'geometry' in out.columns:
        out = out[out.geometry.notna()].copy()
        try:
            out = out[~out.geometry.is_empty].copy()
        except Exception:
            pass
    return out


def site_extent_from_config(config):
    """Return [xmin, xmax, ymin, ymax] from a notebook SITE_CONFIG dict."""
    lon = config['pixel_lon']
    lat = config['pixel_lat']
    buf = config['region_buffer_deg']
    return [lon - buf, lon + buf, lat - buf, lat + buf]


def gdf_bounds(df):
    """Return GeoDataFrame bounds or None."""
    if df is None or df.empty or 'geometry' not in df.columns:
        return None
    try:
        bounds = df.total_bounds
    except Exception:
        return None
    if len(bounds) != 4 or not np.isfinite(bounds).all():
        return None
    return bounds


def bounds_overlap(extent_a, extent_b):
    """Check whether two [xmin, xmax, ymin, ymax] extents overlap."""
    ax0, ax1, ay0, ay1 = extent_a
    bx0, bx1, by0, by1 = extent_b
    return (ax0 <= bx1 and ax1 >= bx0 and ay0 <= by1 and ay1 >= by0)


def shared_disturbance_extent(sensor_dfs, site_config, mode='data', pad_frac=0.06):
    """Get a shared plotting extent for disturbance maps."""
    site_extent = site_extent_from_config(site_config)
    data_bounds = []

    for label, df in sensor_dfs.items():
        bounds = gdf_bounds(df)
        if bounds is None:
            print(f'  {label}: no plottable CCDC geometries found.')
            continue

        minx, miny, maxx, maxy = bounds
        data_extent = [minx, maxx, miny, maxy]
        print(
            f'  {label}: {len(df):,} geometries, '
            f'bounds=[{minx:.4f}, {miny:.4f}, {maxx:.4f}, {maxy:.4f}]'
        )
        if not bounds_overlap(data_extent, site_extent):
            print(
                f'    WARNING: {label} CCDC data do not overlap current SITE_CONFIG region. '
                'This usually means stale cache files or a changed site without rerunning Step 3.'
            )
        data_bounds.append(bounds)

    if mode == 'site' or not data_bounds:
        if not data_bounds:
            print('  No CCDC geometries available; using SITE_CONFIG extent.')
        return site_extent

    data_bounds = np.vstack(data_bounds)
    minx = data_bounds[:, 0].min()
    miny = data_bounds[:, 1].min()
    maxx = data_bounds[:, 2].max()
    maxy = data_bounds[:, 3].max()
    dx = max(maxx - minx, 1e-4)
    dy = max(maxy - miny, 1e-4)
    pad_x = dx * pad_frac
    pad_y = dy * pad_frac
    return [minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y]


def plot_disturbance_map(df, title, year_min, year_max, extent, site_extent=None, cmap='YlOrRd', ax=None):
    """Plot a CCDC year-of-max-change map from a sampled GeoDataFrame."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 7))

    df = clean_disturbance_df(df)
    if df is None or df.empty or 'year_of_max_change' not in df.columns:
        valid = None
    else:
        valid = df.dropna(subset=['year_of_max_change']).copy()
        valid = valid[(valid['year_of_max_change'] >= year_min) & (valid['year_of_max_change'] <= year_max)].copy()

    if valid is not None and not valid.empty:
        valid.plot(
            ax=ax,
            column='year_of_max_change',
            cmap=cmap,
            vmin=year_min,
            vmax=year_max,
            edgecolor='none',
            alpha=0.9,
            legend=False,
        )
        n_changed = len(valid)
    else:
        n_changed = 0
        ax.text(
            0.5, 0.5, 'No CCDC change pixels\nin this map/year window',
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=10,
            color='#546E7A',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#B0BEC5', alpha=0.9),
        )

    if site_extent is not None:
        site_x0, site_x1, site_y0, site_y1 = site_extent
        site_rect = mpatches.Rectangle(
            (site_x0, site_y0),
            site_x1 - site_x0,
            site_y1 - site_y0,
            fill=False,
            edgecolor='#263238',
            linewidth=1.2,
            linestyle='--',
            zorder=5,
        )
        ax.add_patch(site_rect)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_facecolor('#FAFAFA')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.tick_params(labelsize=8)
    ax.grid(False)
    ax.text(
        0.02, 0.97, f'Changed pixels: {n_changed:,}',
        transform=ax.transAxes,
        fontsize=9,
        va='top',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.8),
    )

    if standalone:
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=year_min, vmax=year_max), cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.72, pad=0.02)
        cbar.set_label('Year of max change')
        plt.tight_layout()
        return fig
    return ax


def robust_ts_stats(df, band):
    """Compute simple variability diagnostics from a pixel time series."""
    out = {
        'n_obs': 0,
        'median_gap_days': np.nan,
        'value_mad': np.nan,
        'value_iqr': np.nan,
        'diff_mad': np.nan,
        'yearly_count_cv': np.nan,
    }
    if df is None or df.empty or band not in df.columns:
        return out

    work = df[['date', band]].dropna().copy().sort_values('date').reset_index(drop=True)
    if work.empty:
        return out

    vals = work[band].to_numpy(dtype=float)
    out['n_obs'] = len(work)
    out['value_mad'] = np.median(np.abs(vals - np.median(vals)))
    out['value_iqr'] = np.percentile(vals, 75) - np.percentile(vals, 25)

    if len(work) > 1:
        gaps = work['date'].diff().dt.days.dropna().to_numpy(dtype=float)
        out['median_gap_days'] = np.median(gaps)
        diffs = np.diff(vals)
        out['diff_mad'] = np.median(np.abs(diffs - np.median(diffs)))

    yearly_counts = work.groupby(work['date'].dt.year).size().to_numpy(dtype=float)
    if len(yearly_counts) > 1 and np.mean(yearly_counts) > 0:
        out['yearly_count_cv'] = np.std(yearly_counts, ddof=0) / np.mean(yearly_counts)
    return out


def fragmentation_stats(mask):
    """Compute simple connected-component fragmentation diagnostics."""
    out = {
        'predicted_cells': 0,
        'n_patches': 0,
        'mean_patch_cells': np.nan,
        'median_patch_cells': np.nan,
        'singleton_share': np.nan,
        'largest_patch_share': np.nan,
    }
    if mask is None:
        return out

    arr = np.asarray(mask, dtype=bool)
    out['predicted_cells'] = int(arr.sum())
    if out['predicted_cells'] == 0:
        return out

    visited = np.zeros(arr.shape, dtype=bool)
    patch_sizes = []
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    rows, cols = arr.shape

    for r in range(rows):
        for c in range(cols):
            if not arr[r, c] or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            size = 0
            while q:
                rr, cc = q.popleft()
                size += 1
                for dr, dc in neighbors:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and arr[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            patch_sizes.append(size)

    patch_sizes = np.array(patch_sizes, dtype=float)
    out['n_patches'] = int(len(patch_sizes))
    out['mean_patch_cells'] = float(np.mean(patch_sizes))
    out['median_patch_cells'] = float(np.median(patch_sizes))
    out['singleton_share'] = float(np.mean(patch_sizes == 1))
    out['largest_patch_share'] = float(np.max(patch_sizes) / max(out['predicted_cells'], 1))
    return out


def markdown_table(df):
    """Convert a small DataFrame to a Markdown table."""
    out = df.copy().reset_index()
    out = out.rename(columns={out.columns[0]: 'Sensor'})
    headers = list(out.columns)
    rows = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
    ]
    for _, row in out.iterrows():
        vals = []
        for col in headers:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(f'{val:.1f}')
            else:
                vals.append(str(val))
        rows.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(rows)


__all__ = [
    'COMMON_BANDS',
    'L8_BANDS_IN',
    'MOVING_AVG_MIN_PERIODS',
    'MOVING_AVG_WINDOW_DAYS',
    'S2_BANDS_IN',
    'SEG_COLORS',
    'add_indices',
    'add_smooth_fit',
    'bounds_overlap',
    'build_moving_average_fit',
    'ccdc_array_to_scalar',
    'clean_disturbance_df',
    'count_by_year',
    'count_clear_obs',
    'decimal_to_datetime',
    'fragmentation_stats',
    'gdf_bounds',
    'get_ccdc_coefs',
    'get_dates',
    'get_pixel_ts',
    'harmonic_model',
    'load_hls_collections',
    'load_or_compute_ccdc',
    'markdown_table',
    'mask_hls',
    'plot_band_figure',
    'plot_disturbance_map',
    'prepare_l30',
    'prepare_s30',
    'read_polygon_shapefile',
    'robust_ts_stats',
    'run_ccdc_at_point',
    'scalar_ccdc_to_gdf',
    'shared_disturbance_extent',
    'site_extent_from_config',
    'to_decimal_year',
]
