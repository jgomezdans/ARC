import arc
from pathlib import Path
from osgeo import gdal, osr
import numpy as np
import datetime as dt
import pandas as pd

gdal.UseExceptions()

# Constants
start_date = "2019-01-01"
end_date = "2019-10-15"
START_OF_SEASON = 70
CROP_TYPE = "wheat"
GROWTH_SEASON_LENGTH = 70
NUM_SAMPLES = 100000

LAZY_EVALUATION_STEP = 100
ALPHA = 0.8
LINE_WIDTH = 2
# Set up a "crop map" for each field and year....
data = {
    "year": [
        2019,
        2019,
        2019,
        2019,
        2019,
        2019,
        2020,
        2020,
        2020,
        2020,
        2020,
        2020,
        2021,
        2021,
        2021,
        2021,
        2021,
        2021,
        2022,
        2022,
        2022,
        2022,
        2022,
        2022,
        2024,
        2024,
        2024,
        2024,
        2024,
        2024,
    ],
    "field": [
        1,
        2,
        3,
        4,
        5,
        6,
        1,
        2,
        3,
        4,
        5,
        6,
        1,
        2,
        3,
        4,
        5,
        6,
        1,
        2,
        3,
        4,
        5,
        6,
        1,
        2,
        3,
        4,
        5,
        6,
    ],
    "crop_type": [
        "Wheat",
        "Wheat",
        "Wheat",
        "Maize",
        "Maize",
        "Maize",
        "Maize",
        "Maize",
        "Wheat",
        "Wheat",
        "Wheat",
        "Wheat",
        "Maize",
        "Wheat",
        "Maize",
        "Maize",
        "Maize",
        "Maize",
        "Wheat",
        "Maize",
        "Maize",
        "Maize",
        "Maize",
        "Maize",
        "Wheat",
        "Maize",
        "Wheat",
        "Maize",
        "Maize",
        "Maize",
    ],
}

# Creating the dataframe
cropmap_df = pd.DataFrame(data)


def npz_to_geotiff(npz_dump):
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
    geojson_path: str | Path,
    arc_dir: str | Path = "/home/jose/python/ARC/bulgaria/",
):
    """Main function to execute the Arc field processing and plotting"""
    field_no = int(geojson_path.stem.split("_")[-1])

    for year in [2021, 2022, 2024]:
        start_date = f"{year}-1-01"
        end_date = f"{year}-10-15"

        the_crop = cropmap_df[
            np.logical_and(
                cropmap_df.year == year, cropmap_df.field == field_no
            )
        ].crop_type.values[0]
        if the_crop.lower() == "wheat":
            start_of_season = 30
            season_length = 114
        elif the_crop.lower() == "maize":
            start_of_season = 124
            season_length = 64
        arc_dir = Path(arc_dir)
        suffix = Path(geojson_path).stem
        S2_data_folder = Path(arc_dir) / suffix / "S2_data"
        S2_data_folder.mkdir(parents=True, exist_ok=True)

        output_dump_fname = f"{(Path(arc_dir) / suffix).as_posix()}/{suffix}_{year}_inversion.npz"
        print(output_dump_fname)
        scale_data, post_bio_tensor, post_bio_unc_tensor, mask, doys = (
            arc.arc_field(
                start_date,
                end_date,
                geojson_path,
                start_of_season,
                the_crop.lower(),
                output_dump_fname,
                NUM_SAMPLES,
                season_length,
                str(S2_data_folder),
                plot=False,
            )
        )
        npz_to_geotiff(output_dump_fname)


if __name__ == "__main__":
    geojsons = sorted(
        [
            f
            for f in Path("/home/jose/python/ARC/bulgaria").glob(
                "field_?.geojson"
            )
        ]
    )
    # with ProcessPoolExecutor() as executor:
    for geo in geojsons:
        # executor.submit(main, geo)
        main(geo)
