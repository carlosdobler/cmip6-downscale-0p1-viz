import numpy as np
import pandas as pd
import xarray as xr
import gcsfs
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import gaussian_kde
from shiny import App, reactive, render, ui

# --- Constants ---
BUCKET = "clim_data_reg_useast1"
DISPLAY_CELLS = 60
CACHE_PADDING = 10
MIN_CITY_POP = 500_000
KDE_BANDWIDTH = 0.05

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

    if var_key == "precipitation":
        # Convert ERA5 from meters to mm
        ds_a[cfg["era5_var"]] = ds_a[cfg["era5_var"]] * 1000
        # Convert CMIP6 precipitation from kg m-2 s-1 (mm/s) to mm/day
        ds_b[cfg["cmip6_var"]] = ds_b[cfg["cmip6_var"]] * 86400
        ds_clim[cfg["cmip6_var"]] = ds_clim[cfg["cmip6_var"]] * 86400

        # Recompute relative difference in app: (CMIP6 - ERA5) / ERA5 * 100
        era5_clim_path = (
            f"{BUCKET}/era5_land/climatologies/precipitation_mean_1971-2010.zarr"
        )
        ds_era5_clim = xr.open_zarr(fs.get_mapper(era5_clim_path)).load()
        ds_era5_clim = standardize_dims(ds_era5_clim)

        era5_mm = ds_era5_clim[cfg["era5_var"]] * 1000
        cmip6_mm = ds_clim[cfg["cmip6_var"]]

        diff_val = (cmip6_mm - era5_mm) / era5_mm * 100
        ds_diff[cfg["diff_var"]] = diff_val

    return ds_a, ds_b, ds_clim, ds_diff


# --- UI ---
app_ui = ui.page_fluid(
    ui.h2("Climate Data Comparison Dashboard", style="text-align: center;"),
    # Section 1: Variable and location
    ui.div(
        ui.card(
            ui.row(
                ui.column(
                    4,
                    ui.input_select(
                        "variable",
                        "Climate Variable",
                        {k: v["label"] for k, v in VARS_CONFIG.items()},
                    ),
                ),
                ui.column(3, ui.input_numeric("lat", "Latitude", 46.0, step=0.1)),
                ui.column(3, ui.input_numeric("lon", "Longitude", 8.2, step=0.1)),
                ui.column(2, ui.input_action_button("go", "Go", class_="btn-primary")),
            )
        ),
        style="max-width: 1200px; margin: 0 auto;",
    ),
    # Section 2: Single timestep map
    ui.div(
        ui.card(
            ui.row(
                ui.column(
                    3,
                    ui.input_date("date", "Date", value="1971-01-01", max="2025-12-31"),
                    ui.input_action_button("random_date", "Random date"),
                ),
                ui.column(6, ui.output_plot("plot_timestep", height="600px")),
            )
        ),
        style="max-width: 1200px; margin: 0 auto;",
    ),
    # Section 3: Climatological mean and diff maps
    ui.div(
        ui.card(
            ui.row(
                ui.column(6, ui.output_plot("plot_clim_b", height="600px")),
                ui.column(6, ui.output_plot("plot_diff", height="600px")),
            )
        ),
        style="max-width: 1200px; margin: 0 auto;",
    ),
    # Section 4: Distribution plots
    ui.div(
        ui.card(
            ui.row(
                ui.column(9, ui.output_plot("plot_density", height="400px")),
                ui.column(
                    3,
                    ui.div(
                        ui.input_action_button(
                            "compute_density",
                            "Compute distributions",
                            class_="btn-info w-100",
                        ),
                        ui.div(ui.output_ui("output_stats"), style="margin-top: 20px;"),
                        style="padding-top: 20px;",
                    ),
                ),
            )
        ),
        style="max-width: 1200px; margin: 0 auto;",
    ),
)


# --- Server ---
def server(input, output, session):
    # Reactive values for state
    center = reactive.Value((46.0, 8.2))
    cache_center = reactive.Value(None)
    cache_b = reactive.Value(None)  # Downscaled Clim mean cache
    cache_diff = reactive.Value(None)  # Diff cache

    # Trigger for single timestep map update
    trigger_timestep = reactive.Value(0)

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
        trigger_timestep.set(trigger_timestep.get() + 1)

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
        trigger_timestep.set(trigger_timestep.get() + 1)

    @reactive.Effect
    @reactive.event(input.random_date)
    def _():
        if ds_lazy_b.get() is not None:
            times = ds_lazy_b.get().time.values
            rand_time = np.random.choice(times)
            dt = pd.to_datetime(rand_time)
            ui.update_date("date", value=dt.strftime("%Y-%m-%d"))
            # update_date happens in next tick, but we can trigger update now
            trigger_timestep.set(trigger_timestep.get() + 1)

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

        # Return fixed extent based on center, not data availability
        extent = (
            lat_c - disp_half,
            lat_c + disp_half,
            lon_c - disp_half,
            lon_c + disp_half,
        )

        return crop_b, crop_d, extent

    @reactive.Calc
    def single_timestep():
        trigger_timestep()  # Dependency on trigger
        if ds_lazy_b.get() is None:
            return None

        lat_c, lon_c = center.get()
        sel_date = input.date()

        res = 0.1
        disp_half = DISPLAY_CELLS * res
        eps = res / 2.0

        cfg = VARS_CONFIG[input.variable()]

        da = ds_lazy_b.get()[cfg["cmip6_var"]].sel(time=sel_date, method="nearest")
        da_crop = da.sel(
            lat=slice(lat_c - disp_half - eps, lat_c + disp_half + eps),
            lon=slice(lon_c - disp_half - eps, lon_c + disp_half + eps),
        ).compute()

        # Return fixed extent based on center, not data availability
        extent = (
            lat_c - disp_half,
            lat_c + disp_half,
            lon_c - disp_half,
            lon_c + disp_half,
        )

        return da_crop, da_crop.time.values, extent

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

    def render_map(da, extent, title, vmin, vmax, cmap, cbar_label, units):
        # Use a square figure size to minimize empty space
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.subplots_adjust(left=0.1, right=0.88, top=0.95, bottom=0.05)

        lat_min, lat_max, lon_min, lon_max = extent

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
            return fig

        res = abs(float(da.lat[1] - da.lat[0])) if da.lat.size > 1 else 0.1
        img_lat_min = float(da.lat.min()) - res / 2
        img_lat_max = float(da.lat.max()) + res / 2
        img_lon_min = float(da.lon.min()) - res / 2
        img_lon_max = float(da.lon.max()) + res / 2

        # Use aspect="equal" to ensure 1deg lat == 1deg lon
        im = ax.imshow(
            arr,
            origin="lower",
            extent=[img_lon_min, img_lon_max, img_lat_min, img_lat_max],
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            interpolation="nearest",
        )

        # Enforce perfectly square geographic limits early
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_aspect("equal")

        # Overlay layers: Check for empty and reset aspect after each plot
        # because geopandas .plot() overrides ax.set_aspect internally
        coast_clip = gdf_coast.cx[lon_min:lon_max, lat_min:lat_max]
        if not coast_clip.empty:
            coast_clip.plot(ax=ax, color="#555555", linewidth=0.8)
            ax.set_aspect("equal")

        admin_clip = gdf_admin.cx[lon_min:lon_max, lat_min:lat_max]
        if not admin_clip.empty:
            admin_clip.plot(ax=ax, facecolor="none", edgecolor="#555555", linewidth=0.8)
            ax.set_aspect("equal")

        visible_cities = gdf_cities.cx[lon_min:lon_max, lat_min:lat_max]
        visible_cities = visible_cities[visible_cities["POP_MAX"] >= MIN_CITY_POP]

        for idx, row in visible_cities.iterrows():
            ax.text(
                row.geometry.x,
                row.geometry.y,
                row["NAME"],
                fontsize=9,
                alpha=0.8,
                ha="center",
                va="center",
            )

        lat_c, lon_c = center.get()
        ax.plot(lon_c, lat_c, "r+", markersize=15)

        try:
            val = float(da.sel(lat=lat_c, lon=lon_c, method="nearest").values)
            caption = f"Value at crosshairs: {val:.2f} {units}"
        except Exception:
            caption = "Value at crosshairs: N/A"

        cbar = plt.colorbar(
            im, ax=ax, orientation="vertical", fraction=0.03, pad=0.05, shrink=0.85
        )
        cbar.ax.tick_params(labelsize=9)
        cbar.ax.set_title(cbar_label, fontsize=9, pad=10)

        if title:
            ax.set_title(title, fontsize=12, pad=10)

        ax.tick_params(axis="both", labelsize=9)
        ax.set_xlabel(caption, fontsize=11, labelpad=10)
        ax.set_ylabel("")

        # tight_layout with a slightly larger padding to ensure labels aren't cropped
        fig.tight_layout(pad=3.0)
        return fig

    @output
    @render.plot
    def plot_timestep():
        data = single_timestep()
        if data is None:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Click 'Go' or 'Random date' to show map", ha="center")
            return fig
        da, ts, extent = data
        cfg = VARS_CONFIG[input.variable()]
        return render_map(
            da, extent, "", None, None, "viridis", cfg["units"], cfg["units"]
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
            cfg["units"],
            cfg["units"],
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
        vmax = max(abs(da.min().values), abs(da.max().values))
        return render_map(
            da,
            extent,
            "Difference (Downscaled - ERA5)",
            -vmax,
            vmax,
            "RdBu_r",
            f"Diff ({cfg['diff_units']})",
            cfg["diff_units"],
        )

    @output
    @render.plot
    def plot_density():
        data = center_density()
        if data is None:
            fig, ax = plt.subplots()
            ax.text(
                0.5, 0.5, "Press 'Compute distributions' to show results", ha="center"
            )
            return fig

        ts_a, ts_b, lat_c, lon_c = data
        cfg = VARS_CONFIG[input.variable()]

        ts_a = ts_a[~np.isnan(ts_a)]
        ts_b = ts_b[~np.isnan(ts_b)]

        if len(ts_a) == 0 or len(ts_b) == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data at this location", ha="center")
            return fig

        fig, ax = plt.subplots(figsize=(8, 5))

        if input.variable() == "precipitation":
            for ts, label, color in [
                (ts_a, "ERA5-Land", "#1f77b4"),
                (ts_b, "Downscaled data", "#ff7f0e"),
            ]:
                x = np.sort(ts)
                y = np.arange(len(x), 0, -1) / len(x)
                mask = x >= 1.0
                prob_gt_1 = np.mean(ts > 1.0)

                if np.any(mask):
                    plot_x = np.insert(x[mask], 0, 1.0)
                    plot_y = np.insert(y[mask], 0, prob_gt_1)
                    ax.plot(plot_x, plot_y, label=label, color=color, linewidth=2)
                else:
                    ax.plot([], [], label=label, color=color)

            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
            ax.set_xlabel(f"{cfg['units']}")
            ax.set_ylabel("Probability P(Value > x)")
            ax.set_ylim(bottom=0)
            all_vals = np.concatenate([ts_a, ts_b])
            max_val = np.max(all_vals) if len(all_vals) > 0 else 10
            ax.set_xlim(left=1, right=max(10, max_val * 1.1))
        else:
            x_min, x_max = min(ts_a.min(), ts_b.min()), max(ts_a.max(), ts_b.max())
            x_grid = np.linspace(x_min, x_max, 500)
            kde_a, kde_b = (
                gaussian_kde(ts_a, bw_method=KDE_BANDWIDTH),
                gaussian_kde(ts_b, bw_method=KDE_BANDWIDTH),
            )
            y_a, y_b = kde_a(x_grid), kde_b(x_grid)
            ax.fill_between(x_grid, y_a, alpha=0.4, color="#1f77b4", label="ERA5-Land")
            ax.plot(x_grid, y_a, color="#1f77b4", alpha=0.9)
            ax.fill_between(
                x_grid, y_b, alpha=0.4, color="#ff7f0e", label="Downscaled data"
            )
            ax.plot(x_grid, y_b, color="#ff7f0e", alpha=0.9)
            ax.set_xlabel(f"{cfg['units']}")
            ax.set_ylabel("Density")

        ax.legend()
        return fig

    @output
    @render.ui
    def output_stats():
        data = center_density()
        if data is None:
            return ui.div()

        ts_a, ts_b, _, _ = data
        ts_a = ts_a[~np.isnan(ts_a)]
        ts_b = ts_b[~np.isnan(ts_b)]

        if len(ts_a) == 0 or len(ts_b) == 0:
            return ui.div()

        is_pr = input.variable() == "precipitation"

        if is_pr:
            row1_label = "Dry days (<1mm)"
            row1_a = f"{np.mean(ts_a <= 1.0) * 100:.1f}%"
            row1_b = f"{np.mean(ts_b <= 1.0) * 100:.1f}%"
            row2_label = "Max (mm/d)"
            row2_a = f"{np.max(ts_a):.1f}"
            row2_b = f"{np.max(ts_b):.1f}"
        else:
            row1_label = "Min"
            row1_a = f"{np.min(ts_a):.1f}"
            row1_b = f"{np.min(ts_b):.1f}"
            row2_label = "Max"
            row2_a = f"{np.max(ts_a):.1f}"
            row2_b = f"{np.max(ts_b):.1f}"

        return ui.HTML(
            f"""
            <table class="table table-sm table-bordered" style="font-size: 0.85rem;">
                <thead>
                    <tr class="table-light">
                        <th>Metric</th>
                        <th>ERA5</th>
                        <th>Down.</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>{row1_label}</strong></td>
                        <td>{row1_a}</td>
                        <td>{row1_b}</td>
                    </tr>
                    <tr>
                        <td><strong>{row2_label}</strong></td>
                        <td>{row2_a}</td>
                        <td>{row2_b}</td>
                    </tr>
                </tbody>
            </table>
        """
        )


app = App(app_ui, server)
