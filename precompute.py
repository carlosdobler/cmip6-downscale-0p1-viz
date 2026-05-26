import xarray as xr
import gcsfs
import dask
from dask.distributed import Client, LocalCluster
import cartopy.io.shapereader as shpreader
import geopandas as gpd
import time
import os

# --- Constants & Paths ---
BUCKET = "gs://clim_data_reg_useast1"
ERA5_PATH = f"{BUCKET}/era5_land"
CMIP6_PATH = f"{BUCKET}/cmip6_downscaled_woodwell"
CMIP6_MODEL = "MPI-ESM1-2-HR_ww-isimip_ssp585"


def _to_kebab(s):
    return s.replace("_", "-")


def _make_var_config(era5_var, cmip6_var):
    era5_var_kebab = _to_kebab(era5_var)
    return {
        "era5_var": era5_var,
        "cmip6_var": cmip6_var,
        "era5_path": f"{ERA5_PATH}/daily_aggregates/{era5_var}.zarr",
        "cmip6_path": f"{CMIP6_PATH}/daily/{cmip6_var}/{cmip6_var}_{CMIP6_MODEL}_day.zarr",
        "out_era5": f"{ERA5_PATH}/climatologies/{era5_var_kebab}_mean_1971-2010.zarr",
        "out_cmip6": f"{CMIP6_PATH}/climatologies/{cmip6_var}/{cmip6_var}_{CMIP6_MODEL}_mean_1971-2010.zarr",
    }


VARS_CONFIG = {
    "tas": _make_var_config("temperature_2m", "tas"),
    "tasmax": _make_var_config("temperature_2m_max", "tasmax"),
    "tasmin": _make_var_config("temperature_2m_min", "tasmin"),
    "pr": _make_var_config("total_precipitation_sum", "pr"),
    "hurs": _make_var_config("relative_humidity", "hurs"),
    "rsds": _make_var_config("surface_solar_radiation_downwards_sum", "rsds"),
    "sfcwind": {
        **_make_var_config("wind_speed_10m", "sfcwind"),
        "era5_var": "wind_speed",
    },
}

CLIM_START = "1971-01-01"
CLIM_END = "2010-12-31"

# SHAPEFILE_PATHS = {
#     "coastline": f"{BUCKET}/misc_data/physical/coastline.parquet",
#     "admin": f"{BUCKET}/misc_data/admin_units/admin_0_countries.parquet",
#     "population": f"{BUCKET}/misc_data/population/populated_places.parquet",
# }


def standardize_dims(ds):
    """Standardize dimension names to lat/lon."""
    rename_dict = {}
    if "y" in ds.dims:
        rename_dict["y"] = "lat"
    if "x" in ds.dims:
        rename_dict["x"] = "lon"
    if "latitude" in ds.dims:
        rename_dict["latitude"] = "lat"
    if "longitude" in ds.dims:
        rename_dict["longitude"] = "lon"
    return ds.rename(rename_dict)


def precompute_climatologies():
    fs = gcsfs.GCSFileSystem()

    for var_name, cfg in VARS_CONFIG.items():
        print(f"--- Processing variable: {var_name} ---")
        start_time = time.time()

        # 1. Open Zarr stores lazily
        print(f"Opening Zarr stores for {var_name}...")
        ds_era5 = xr.open_zarr(
            fs.get_mapper(cfg["era5_path"]), chunks="auto", consolidated=False
        )
        ds_cmip6 = xr.open_zarr(
            fs.get_mapper(cfg["cmip6_path"]), chunks="auto", consolidated=False
        )

        ds_era5 = standardize_dims(ds_era5)
        ds_cmip6 = standardize_dims(ds_cmip6)

        # 2. Select 40-year period and compute mean
        print(f"Computing climatological mean (1971-2010) for {var_name}...")
        era5_da = ds_era5[cfg["era5_var"]].sel(time=slice(CLIM_START, CLIM_END))
        cmip6_da = ds_cmip6[cfg["cmip6_var"]].sel(time=slice(CLIM_START, CLIM_END))

        # # Handle units for precipitation
        # if var_name == "pr":
        #     print("Converting precipitation units...")
        #     # ERA5 total_precipitation_sum is in meters, convert to mm/day
        #     era5_da = era5_da * 1000
        #     # CMIP6 pr is in mm/s (kg m-2 s-1), convert to mm/day
        #     cmip6_da = cmip6_da * 86400

        # elif var_name == "rsds":
        #     print("Converting rsds units (J/m2 -> W/m2)...")
        #     # ERA5 is daily sum in J/m2. Divide by seconds in a day to get W/m2 (J/s/m2).
        #     era5_da = era5_da / 86400

        era5_clim = era5_da.mean(dim="time")
        cmip6_clim = cmip6_da.mean(dim="time")

        # 3. Save resulting 2D arrays back to GCS
        # Output as single chunk for faster load in Shiny app
        print(f"Saving ERA5 climatology to {cfg['out_era5']}...")
        era5_clim = era5_clim.chunk({"lat": -1, "lon": -1})
        era5_clim.to_dataset().to_zarr(
            fs.get_mapper(cfg["out_era5"]), mode="w", consolidated=True
        )

        print(f"Saving CMIP6 climatology to {cfg['out_cmip6']}...")
        cmip6_clim = cmip6_clim.chunk({"lat": -1, "lon": -1})
        cmip6_clim.to_dataset().to_zarr(
            fs.get_mapper(cfg["out_cmip6"]), mode="w", consolidated=True
        )

        end_time = time.time()
        print(f"Finished {var_name} in {end_time - start_time:.2f} seconds.")


# def precompute_shapefiles():
#     print("--- Downloading and preparing shapefiles ---")
#     fs = gcsfs.GCSFileSystem()

#     # Coastlines
#     print("Downloading Natural Earth coastlines...")
#     coast_path = shpreader.natural_earth(
#         resolution="50m", category="physical", name="coastline"
#     )
#     coast_gdf = gpd.read_file(coast_path)
#     print(f"Saving coastlines to {SHAPEFILE_PATHS['coastline']}...")
#     coast_gdf.to_parquet(SHAPEFILE_PATHS["coastline"], filesystem=fs)

#     # Country borders
#     print("Downloading Natural Earth country borders...")
#     admin_path = shpreader.natural_earth(
#         resolution="50m", category="cultural", name="admin_0_countries"
#     )
#     admin_gdf = gpd.read_file(admin_path)
#     print(f"Saving country borders to {SHAPEFILE_PATHS['admin']}...")
#     admin_gdf.to_parquet(SHAPEFILE_PATHS["admin"], filesystem=fs)

#     # Populated places
#     print("Downloading Natural Earth populated places...")
#     pop_path = shpreader.natural_earth(
#         resolution="50m", category="cultural", name="populated_places"
#     )
#     pop_gdf = gpd.read_file(pop_path)
#     pop_gdf_filtered = pop_gdf[["NAME", "geometry", "POP_MAX"]]
#     print(f"Saving populated places to {SHAPEFILE_PATHS['population']}...")
#     pop_gdf_filtered.to_parquet(SHAPEFILE_PATHS["population"], filesystem=fs)


if __name__ == "__main__":
    # Start Dask Distributed Client for true parallel processing across CPUs
    cluster = LocalCluster()
    client = Client(cluster)
    print(f"Dask dashboard available at: {client.dashboard_link}")

    try:
        precompute_climatologies()
        # precompute_shapefiles()
        print("Precomputation complete.")
    finally:
        client.close()
        cluster.close()
