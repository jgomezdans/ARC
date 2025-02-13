import arc
from pathlib import Path
from osgeo import gdal, osr
import numpy as np
import datetime as dt
import shapely.geometry
import geopandas as gpd

gdal.UseExceptions()
# Constants
START_OF_SEASON = 70
CROP_TYPE = "wheat"
GROWTH_SEASON_LENGTH = 70
NUM_SAMPLES = 100000

BG_CROP_CALENDAR = {"wheat": (30, 114), "maize": (124, 64)}


def npz_to_geotiff(npz_dump: str) -> None:
    """
    Converts a .npz file containing numpy arrays to a GeoTIFF file.

    Parameters:
    npz_dump (str): Path to the .npz file containing the numpy arrays.

    Returns:
    None
    """
    params = ["N", "Cab", "Cm", "Cw", "LAI", "ALA", "Cbrown"]
    scaling = [
        1.0 / 100.0,
        1.0 / 100.0,
        1 / 10000.0,
        1 / 10000.0,
        1 / 100.0,
        1 / 100.0,
        1 / 1000.0,
    ]

    npz_dump = Path(npz_dump)
    the_year = npz_dump.stem.split("_")[-2]
    f = np.load(npz_dump)
    geotransform = f.f.geotransform
    crs = str(f.f.crs)
    mask = f.f.mask * 1.0
    output_raster = np.zeros_like(mask) * np.nan
    param_est = f.f.post_bio_tensor
    _, _, nt = param_est.shape
    ny, nx = mask.shape
    dates = [
        dt.datetime.strptime(f"{the_year}/{s:03d}", "%Y/%j") for s in f.f.doys
    ]
    for i, param in enumerate(params):
        out_fname = (
            npz_dump.parents[0] / f"{npz_dump.stem}_{param}_{the_year}.tif"
        )
        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(out_fname, nx, ny, nt, gdal.GDT_Float32)
        out_ds.SetGeoTransform(geotransform)
        srs = osr.SpatialReference()
        srs.ImportFromWkt(crs)
        out_ds.SetProjection(srs.ExportToWkt())
        for j in range(nt):
            banda = out_ds.GetRasterBand(j + 1)
            output_raster = np.zeros_like(mask) * np.nan
            output_raster[mask == 0.0] = param_est[:, i, j] * scaling[i]
            banda.WriteArray(output_raster.astype(np.float32))
            banda.SetMetadata({"DoY": f.f.doys[j], "Date": dates[j]})
        out_ds = None


def main(
    year: int,
    crop: str,
    field_no: int,
    field_geometry: str | Path | shapely.geometry.base.BaseGeometry,
    arc_dir: str | Path = "/home/jose/python/ARC/bulgaria/",
):
    """Main function to execute the Arc field processing and plotting"""

    start_date = f"{year}-01-01"
    end_date = f"{year}-10-15"

    arc_dir = Path(arc_dir)
    S2_data_folder = Path(arc_dir) / "S2_data"
    S2_data_folder.mkdir(parents=True, exist_ok=True)

    output_dump_fname = f"{Path(arc_dir).as_posix()}/field_{crop}_{field_no}_{year}_inversion.npz"
    print(output_dump_fname)
    scale_data, post_bio_tensor, post_bio_unc_tensor, mask, doys = (
        arc.arc_field(
            start_date,
            end_date,
            field_geometry,
            BG_CROP_CALENDAR[crop][0],
            crop.lower(),
            output_dump_fname,
            NUM_SAMPLES,
            BG_CROP_CALENDAR[crop][1],
            str(S2_data_folder),
            plot=False,
        )
    )
    npz_to_geotiff(output_dump_fname)


if __name__ == "__main__":
    # start_date = "2019-01-01"0
    # end_date = "2019-10-15"

    # geojsons = sorted(
    #     [
    #         f
    #         for f in Path("/home/jose/python/ARC/bulgaria").glob(
    #             "field_?.geojson"
    #         )
    #     ]
    # )
    # # with ProcessPoolExecutor() as executor:
    # for geo in geojsons:
    #     # executor.submit(main, geo)
    #     main(geo)

    # df = gpd.read_file(field_location_file)
    # df = df.to_crs(epsg=4326)  # Convert to WGS84
    # for idx, row in df.iterrow():
    #     main(row["year"], "wheat", row["geometry"])
    geojson_file = "/home/jose/python/ARC/bulgaria/bulgaria_fields.geojson"
    gdf = gpd.read_file(geojson_file)
    gdf = gdf.to_crs(epsg=4326)  # Convert to WGS84
    for idx, row in gdf.iterrows():
        main(2017, "wheat", row["Field_N"], row["geometry"])
