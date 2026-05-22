import numpy as np
import pandas as pd
import xarray as xr
import gcsfs
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from shiny import App, reactive, render, ui

# --- Constants ---
BUCKET = "clim_data_reg_useast1"
DISPLAY_CELLS = 40
CACHE_PADDING = 10
MIN_CITY_POP = 500_000
KDE_BANDWIDTH = 0.3

# --- Variable configuration ---
VARS_CONFIG = {
    "temperature": {
        "label": "Mean Temperature",
        "units": "K",
        "diff_units": "K",
        "diff_type": "absolute",
        "era5_path": f"{BUCKET}/era5_land/daily_aggregates/temperature_2m.zarr",
        "cmip6_path": f"{BUCKET}/cmip6_downscaled_woodwell/daily/tas/tas_MPI-ESM1-2-HR_ww-isimip_ssp585_day.zarr",
        "clim_path": f"{BUCKET}/cmip6_downscaled_woodwell/climatologies/temperature/temperature_MPI-ESM1-2-HR_ww-isimip_ssp585_mean_1971-2010.zarr",
        "diff_path": f"{BUCKET}/cmip6_downscaled_woodwell/climatologies_diff_era5land/temperature/",
        "era5_var": "temperature_2m",
        "cmip6_var": "tas",
        "diff_var": "diff",
    },
    "precipitation": {
        "label": "Precipitation",
        "units": "mm/day",
        "diff_units": "%",
        "diff_type": "relative",
        "era5_path": f"{BUCKET}/era5_land/daily_aggregates/total_precipitation_sum.zarr",
        "cmip6_path": f"{BUCKET}/cmip6_downscaled_woodwell/daily/pr/pr_MPI-ESM1-2-HR_ww-isimip_ssp585_day.zarr",
        "clim_path": f"{BUCKET}/cmip6_downscaled_woodwell/climatologies/precipitation/precipitation_MPI-ESM1-2-HR_ww-isimip_ssp585_mean_1971-2010.zarr",
        "diff_path": f"{BUCKET}/cmip6_downscaled_woodwell/climatologies_diff_era5land/precipitation/",
        "era5_var": "total_precipitation_sum",
        "cmip6_var": "pr",
        "diff_var": "diff",
    },
}

SHAPEFILE_PATHS = {
    "coastline": f"{BUCKET}/misc_data/physical/coastline.parquet",
    "admin": f"{BUCKET}/misc_data/admin_units/admin_0_countries.parquet",
    "population": f"{BUCKET}/misc_data/population/populated_places.parquet",
}

# --- Startup (Global Context) ---
fs = gcsfs.GCSFileSystem()

# Load shapefiles once
gdf_coast = gpd.read_parquet(SHAPEFILE_PATHS["coastline"], filesystem=fs)
gdf_admin = gpd.read_parquet(SHAPEFILE_PATHS["admin"], filesystem=fs)
gdf_cities = gpd.read_parquet(SHAPEFILE_PATHS["population"], filesystem=fs)


def standardize_dims(ds):
    """Standardize dimension names to lat/lon and ensure increasing latitude."""
    rename_dict = {}
    if "y" in ds.dims:
        rename_dict["y"] = "lat"
    if "x" in ds.dims:
        rename_dict["x"] = "lon"
    if "latitude" in ds.dims:
        rename_dict["latitude"] = "lat"
    if "longitude" in ds.dims:
        rename_dict["longitude"] = "lon"
    ds = ds.rename(rename_dict)
    # Ensure latitudes are increasing for correct slicing
    if "lat" in ds.coords and ds.lat[0] > ds.lat[-1]:
        ds = ds.sortby("lat")
    return ds


def load_datasets(var_key):
    cfg = VARS_CONFIG[var_key]

    # Load climatology fully into memory
    ds_clim = xr.open_zarr(fs.get_mapper(cfg["clim_path"])).load()
    ds_diff = xr.open_zarr(fs.get_mapper(cfg["diff_path"])).load()

    # Standardize memory datasets
    ds_clim = standardize_dims(ds_clim)
    ds_diff = standardize_dims(ds_diff)

    # Lazy open daily stores (A and B)
    ds_a = xr.open_zarr(fs.get_mapper(cfg["era5_path"]), chunks="auto")
    ds_b = xr.open_zarr(fs.get_mapper(cfg["cmip6_path"]), chunks="auto")

    # Standardize dimensions and coordinates
    ds_a = standardize_dims(ds_a)
    ds_b = standardize_dims(ds_b)

    # Subset to 1971-2025
    ds_a = ds_a.sel(time=slice("1971-01-01", "2025-12-31"))
    ds_b = ds_b.sel(time=slice("1971-01-01", "2025-12-31"))

    return ds_a, ds_b, ds_clim, ds_diff


# --- UI ---
app_ui = ui.page_fluid(
    ui.h2("Climate Data Comparison Dashboard"),
    # Section 1: Variable and location
    ui.card(
        ui.row(
            ui.column(
                3,
                ui.input_select(
                    "variable",
                    "Climate Variable",
                    {k: v["label"] for k, v in VARS_CONFIG.items()},
                ),
            ),
            ui.column(2, ui.input_numeric("lat", "Latitude", 0.0, step=0.1)),
            ui.column(2, ui.input_numeric("lon", "Longitude", 0.0, step=0.1)),
            ui.column(2, ui.input_action_button("go", "Go", class_="btn-primary")),
            ui.column(
                3,
                ui.div(
                    ui.row(
                        ui.column(
                            6, ui.input_action_button("up", "↑", class_="btn-sm")
                        ),
                        ui.column(
                            6, ui.input_action_button("down", "↓", class_="btn-sm")
                        ),
                    ),
                    ui.row(
                        ui.column(
                            6, ui.input_action_button("left", "←", class_="btn-sm")
                        ),
                        ui.column(
                            6, ui.input_action_button("right", "→", class_="btn-sm")
                        ),
                    ),
                    style="width: 100px;",
                ),
            ),
        )
    ),
    # Section 2: Single timestep map
    ui.card(
        ui.row(
            ui.column(
                3,
                ui.input_date("date", "Date", value="2010-08-01", max="2025-12-31"),
                ui.input_action_button("random_date", "Random date"),
                ui.hr(),
                ui.input_action_button(
                    "load_timestep", "Load map", class_="btn-success"
                ),
            ),
            ui.column(6, ui.output_plot("plot_timestep", height="500px")),
        )
    ),
    # Section 3: Climatological mean and diff maps
    ui.card(
        ui.row(
            ui.column(6, ui.output_plot("plot_clim_b", height="500px")),
            ui.column(6, ui.output_plot("plot_diff", height="500px")),
        )
    ),
    # Section 4: Density plots
    ui.card(
        ui.row(
            ui.column(9, ui.output_plot("plot_density", height="400px")),
            ui.column(
                3,
                ui.input_action_button(
                    "compute_density", "Compute densities", class_="btn-info"
                ),
            ),
        )
    ),
)


# --- Server ---
def server(input, output, session):
    # Reactive values for state
    center = reactive.Value((0.0, 0.0))
    cache_center = reactive.Value(None)
    cache_b = reactive.Value(None)  # Downscaled Clim mean cache
    cache_diff = reactive.Value(None)  # Diff cache

    # Currently loaded date for Section 2
    loaded_date = reactive.Value(None)

    # Current active datasets (lazy and pre-loaded)
    ds_lazy_a = reactive.Value(None)
    ds_lazy_b = reactive.Value(None)
    ds_full_clim = reactive.Value(None)
    ds_full_diff = reactive.Value(None)

    # Initial load and variable changes
    @reactive.Effect
    @reactive.event(input.variable, ignore_none=False)
    def _():
        a, b, clim, diff = load_datasets(input.variable())
        ds_lazy_a.set(a)
        ds_lazy_b.set(b)
        ds_full_clim.set(clim)
        ds_full_diff.set(diff)
        update_cache(center.get())

    def update_cache(new_center):
        lat_c, lon_c = new_center

        res = 0.1
        total_cells = DISPLAY_CELLS + CACHE_PADDING
        half = total_cells * res
        # Pad by half a cell so slice() always includes the boundary grid points
        eps = res / 2.0

        c_b = ds_full_clim.get().sel(
            lat=slice(lat_c - half - eps, lat_c + half + eps),
            lon=slice(lon_c - half - eps, lon_c + half + eps),
        )
        c_d = ds_full_diff.get().sel(
            lat=slice(lat_c - half - eps, lat_c + half + eps),
            lon=slice(lon_c - half - eps, lon_c + half + eps),
        )

        cache_b.set(c_b)
        cache_diff.set(c_d)
        cache_center.set(new_center)
        center.set(new_center)

    @reactive.Effect
    @reactive.event(input.go)
    def _():
        update_cache((input.lat(), input.lon()))

    @reactive.Effect
    @reactive.event(input.random_date)
    def _():
        if ds_lazy_b.get() is not None:
            times = ds_lazy_b.get().time.values
            rand_time = np.random.choice(times)
            dt = pd.to_datetime(rand_time)
            ui.update_date("date", value=dt.strftime("%Y-%m-%d"))

    @reactive.Effect
    @reactive.event(input.load_timestep)
    def _():
        loaded_date.set(input.date())

    # Arrow logic
    def move_center(dlat, dlon):
        res = 0.1
        curr_lat, curr_lon = center.get()
        new_lat, new_lon = curr_lat + dlat * res, curr_lon + dlon * res

        # Check if new display window fits in cache with at least one cell margin
        cache_lat, cache_lon = cache_center.get()
        margin = 1 * res
        disp_half = DISPLAY_CELLS * res

        cache_lat_min = cache_lat - (DISPLAY_CELLS + CACHE_PADDING) * res
        cache_lat_max = cache_lat + (DISPLAY_CELLS + CACHE_PADDING) * res
        cache_lon_min = cache_lon - (DISPLAY_CELLS + CACHE_PADDING) * res
        cache_lon_max = cache_lon + (DISPLAY_CELLS + CACHE_PADDING) * res

        if (
            new_lat - disp_half >= cache_lat_min + margin
            and new_lat + disp_half <= cache_lat_max - margin
            and new_lon - disp_half >= cache_lon_min + margin
            and new_lon + disp_half <= cache_lon_max - margin
        ):
            center.set((new_lat, new_lon))
            ui.update_numeric("lat", value=new_lat)
            ui.update_numeric("lon", value=new_lon)
        else:
            update_cache((new_lat, new_lon))
            ui.update_numeric("lat", value=new_lat)
            ui.update_numeric("lon", value=new_lon)

    @reactive.Effect
    @reactive.event(input.up)
    def _():
        move_center(1, 0)

    @reactive.Effect
    @reactive.event(input.down)
    def _():
        move_center(-1, 0)

    @reactive.Effect
    @reactive.event(input.left)
    def _():
        move_center(0, -1)

    @reactive.Effect
    @reactive.event(input.right)
    def _():
        move_center(0, 1)

    @reactive.Calc
    def display_crop():
        if cache_b.get() is None:
            return None
        lat_c, lon_c = center.get()
        res = 0.1
        disp_half = DISPLAY_CELLS * res
        eps = res / 2.0

        crop_b = cache_b.get().sel(
            lat=slice(lat_c - disp_half - eps, lat_c + disp_half + eps),
            lon=slice(lon_c - disp_half - eps, lon_c + disp_half + eps),
        )
        crop_d = cache_diff.get().sel(
            lat=slice(lat_c - disp_half - eps, lat_c + disp_half + eps),
            lon=slice(lon_c - disp_half - eps, lon_c + disp_half + eps),
        )

        lat_min = float(crop_b.lat.min())
        lat_max = float(crop_b.lat.max())
        lon_min = float(crop_b.lon.min())
        lon_max = float(crop_b.lon.max())

        return crop_b, crop_d, (lat_min, lat_max, lon_min, lon_max)

    @reactive.Calc
    def single_timestep():
        if ds_lazy_b.get() is None or loaded_date.get() is None:
            return None

        lat_c, lon_c = center.get()
        sel_date = loaded_date.get()

        res = 0.1
        disp_half = DISPLAY_CELLS * res
        eps = res / 2.0

        cfg = VARS_CONFIG[input.variable()]

        da = ds_lazy_b.get()[cfg["cmip6_var"]].sel(time=sel_date, method="nearest")
        da_crop = da.sel(
            lat=slice(lat_c - disp_half - eps, lat_c + disp_half + eps),
            lon=slice(lon_c - disp_half - eps, lon_c + disp_half + eps),
        ).compute()

        lat_min = float(da_crop.lat.min())
        lat_max = float(da_crop.lat.max())
        lon_min = float(da_crop.lon.min())
        lon_max = float(da_crop.lon.max())

        return da_crop, da_crop.time.values, (lat_min, lat_max, lon_min, lon_max)

    @reactive.Calc
    @reactive.event(input.compute_density)
    def center_density():
        if ds_lazy_a.get() is None:
            return None
        lat_c, lon_c = center.get()
        cfg = VARS_CONFIG[input.variable()]

        ts_a = (
            ds_lazy_a.get()[cfg["era5_var"]]
            .sel(lat=lat_c, lon=lon_c, method="nearest")
            .compute()
        )
        ts_b = (
            ds_lazy_b.get()[cfg["cmip6_var"]]
            .sel(lat=lat_c, lon=lon_c, method="nearest")
            .compute()
        )

        return ts_a.values, ts_b.values, lat_c, lon_c

    def render_map(da, extent, title, vmin, vmax, cmap, var_name):
        fig, ax = plt.subplots(figsize=(12, 9))
        lat_min, lat_max, lon_min, lon_max = extent

        # Ensure lat is ascending and dimensions are in (lat, lon) order before
        # converting to numpy. If the zarr stores dimensions in a different order
        # (e.g. lon, lat) or with descending lat, imshow gets a shape mismatch
        # against the declared extent — matplotlib then computes a zero or negative
        # image height and raises "aspect must be finite and positive".
        if "lat" in da.dims and da.lat[0] > da.lat[-1]:
            da = da.sortby("lat")
        if da.dims != ("lat", "lon"):
            da = da.transpose("lat", "lon")

        arr = np.array(da)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
            ax.text(
                0.5,
                0.5,
                "No data in this region",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(title, fontsize=14)
            return fig

        # Derive imshow extent from the DataArray's actual coordinates,
        # expanded by half a cell so the image fills edge-to-edge rather than
        # center-to-center.  This is what prevents a degenerate (zero-height)
        # image when the coordinate range is very small.
        res = abs(float(da.lat[1] - da.lat[0])) if da.lat.size > 1 else 0.1
        img_lat_min = float(da.lat.min()) - res / 2
        img_lat_max = float(da.lat.max()) + res / 2
        img_lon_min = float(da.lon.min()) - res / 2
        img_lon_max = float(da.lon.max()) + res / 2

        im = ax.imshow(
            arr,
            origin="lower",
            extent=[img_lon_min, img_lon_max, img_lat_min, img_lat_max],
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )

        # Set aspect BEFORE geopandas .plot() calls.
        # Geopandas internally calls ax.set_aspect(1 / cos(y_mean * pi/180)).
        # When the clipped GeoDataFrame is empty, y_mean is nan → crash.
        # Pre-setting "auto" is overridden by geopandas, so we also need
        # the empty-check guards below.
        ax.set_aspect("auto")

        coast_clip = gdf_coast.cx[lon_min:lon_max, lat_min:lat_max]
        if not coast_clip.empty:
            coast_clip.plot(ax=ax, color="#555555", linewidth=0.8)
            ax.set_aspect("auto")

        admin_clip = gdf_admin.cx[lon_min:lon_max, lat_min:lat_max]
        if not admin_clip.empty:
            admin_clip.plot(ax=ax, facecolor="none", edgecolor="#555555", linewidth=0.8)
            ax.set_aspect("auto")

        visible_cities = gdf_cities.cx[lon_min:lon_max, lat_min:lat_max]
        visible_cities = visible_cities[visible_cities["POP_MAX"] >= MIN_CITY_POP]

        for idx, row in visible_cities.iterrows():
            ax.plot(row.geometry.x, row.geometry.y, "k.", markersize=1)
            ax.text(row.geometry.x, row.geometry.y, row["NAME"], fontsize=7, alpha=0.8)

        # Crosshair
        lat_c, lon_c = center.get()
        ax.plot(lon_c, lat_c, "r+", markersize=15)

        plt.colorbar(im, ax=ax, label=f"{var_name}", fraction=0.046, pad=0.04)
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelbottom=False, labelleft=False)
        return fig

    @output
    @render.plot
    def plot_timestep():
        data = single_timestep()
        if data is None:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Select a date and click 'Load map'", ha="center")
            return fig
        da, ts, extent = data
        cfg = VARS_CONFIG[input.variable()]
        dt_str = pd.to_datetime(ts).strftime("%Y-%m-%d")
        return render_map(
            da,
            extent,
            f"Downscaled data — {dt_str}",
            None,
            None,
            "viridis",
            cfg["label"],
        )

    @output
    @render.plot
    def plot_clim_b():
        data = display_crop()
        if data is None:
            return None
        crop_b, _, extent = data
        cfg = VARS_CONFIG[input.variable()]
        da = crop_b[cfg["cmip6_var"]]
        return render_map(
            da,
            extent,
            "Downscaled Climatology (1971-2010)",
            None,
            None,
            "viridis",
            cfg["label"],
        )

    @output
    @render.plot
    def plot_diff():
        data = display_crop()
        if data is None:
            return None
        _, crop_d, extent = data
        cfg = VARS_CONFIG[input.variable()]
        da = crop_d[cfg["diff_var"]]
        # Symmetric divergent range for diff
        vmax = max(abs(da.min().values), abs(da.max().values))
        return render_map(
            da,
            extent,
            "Difference (ERA5 - Downscaled)",
            -vmax,
            vmax,
            "RdBu_r",
            f"Diff ({cfg['units']})",
        )

    @output
    @render.plot
    def plot_density():
        data = center_density()
        if data is None:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Press 'Compute densities' to show results", ha="center")
            return fig

        ts_a, ts_b, lat_c, lon_c = data
        cfg = VARS_CONFIG[input.variable()]

        # Remove NaNs if any
        ts_a = ts_a[~np.isnan(ts_a)]
        ts_b = ts_b[~np.isnan(ts_b)]

        if len(ts_a) == 0 or len(ts_b) == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data at this location", ha="center")
            return fig

        fig, ax = plt.subplots(figsize=(8, 5))

        x_min = min(ts_a.min(), ts_b.min())
        x_max = max(ts_a.max(), ts_b.max())
        x_grid = np.linspace(x_min, x_max, 500)

        kde_a = gaussian_kde(ts_a, bw_method=KDE_BANDWIDTH)
        kde_b = gaussian_kde(ts_b, bw_method=KDE_BANDWIDTH)

        y_a = kde_a(x_grid)
        y_b = kde_b(x_grid)

        ax.fill_between(x_grid, y_a, alpha=0.4, color="#1f77b4", label="ERA5-Land")
        ax.plot(x_grid, y_a, color="#1f77b4", alpha=0.9)

        ax.fill_between(
            x_grid, y_b, alpha=0.4, color="#ff7f0e", label="Downscaled data"
        )
        ax.plot(x_grid, y_b, color="#ff7f0e", alpha=0.9)

        ax.set_title(f"Density at ({lat_c:.2f}, {lon_c:.2f})")
        ax.set_xlabel(f"{cfg['label']} ({cfg['units']})")
        ax.set_ylabel("Density")
        ax.legend()
        return fig


app = App(app_ui, server)
