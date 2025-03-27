from pathlib import Path
from osgeo import gdal
from osgeo import osr
import numpy as np
import datetime as dt


gdal.UseExceptions()
gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")

sel_bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]


def scan_folder_siac(siac_folder: Path | list[Path]) -> dict[dt.datetime, Path]:
    """
    Scans a folder (or a list of folders for  Sentinel 2 SAFE files that
    have already been processed using SIAC. It returns the list of files
    as datetime-index dictionary
    """

    def get_granule(p):
        return next(
            iter(Path(p).glob(f"GRANULE/L1C_{Path(p).name.split('_')[-2]}_A*")), None
        )

    siac_folder = "/home/jose/data/Bulgaria"
    if not isinstance(siac_folder, list):
        siac_folder = [
            siac_folder,
        ]
    file_list = {}
    for folder in siac_folder:
        the_files = sorted(
            [
                f
                for f in Path(folder).glob("*.SAFE")
                if (f / "siac_output.json").exists()
            ]
        )
        file_list.update(
            {
                dt.datetime.strptime(
                    f.stem.split("_")[2], "%Y%m%dT%H%M%S"
                ): get_granule(f)
                for f in the_files
            }
        )
    file_list = {k: v for k, v in file_list.items() if v is not None}
    return file_list


def raa_pixel_function(
    in_ar,
    out_ar,
    xoff,
    yoff,
    xsize,
    ysize,
    raster_xsize,
    raster_ysize,
    buf_radius,
    gt,
    **kwargs,
):
    """Custom pixel function to compute RAA."""
    vaa = in_ar[0] / 1000  # Convert to degrees
    saa = in_ar[1] / 1000  # Convert to degrees
    out_ar[:] = (vaa - saa) % 360


def generate_angle_vrts(
    vaa_vza_file: str, saa_sza_file: str, polygon_wkt: str
) -> tuple:
    """
    Extracts SZA, VZA, and computes RAA from Sentinel-2 angle files.

    Ensures all outputs have a resolution of 10m and clips using a polygon.

    :param vaa_vza_file: Path to VAA_VZA.tif (10m View Azimuth & Zenith)
    :param saa_sza_file: Path to SAA_SZA.tif (5000m Solar Azimuth & Zenith)
    :param polygon_wkt: WKT string of the polygon cutline.
    :return: Tuple with paths to in-memory VRTs for SZA, VZA, and RAA
    """
    # Load raster spatial reference
    raster_ds = gdal.Open(vaa_vza_file)
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster_ds.GetProjection())

    # Define in-memory VRT paths
    sza_vrt = "/vsimem/SZA.vrt"
    vza_vrt = "/vsimem/VZA.vrt"
    raa_vrt = "/vsimem/RAA.vrt"

    # First, resample SAA_SZA file to 10m resolution
    resampled_saa_sza = "/vsimem/SAA_SZA_10m.vrt"
    gdal.Warp(
        resampled_saa_sza,
        saa_sza_file,
        xRes=10,
        yRes=10,
        resampleAlg=gdal.GRA_Bilinear,
        multithread=True,
        dstNodata=np.nan,
        outputType=gdal.GDT_Float32,
    )

    # Then, apply the cutline to the resampled dataset
    #   Try/except blocks are used to handle different projections.
    #   First assume cutline is same as tiff projection, if it moans,
    #   use EPSG:4326 as a fallback.
    # Resample and clip SAA_SZA file
    clipped_saa_sza = "/vsimem/SAA_SZA_clipped.tif"
    try:
        sza_ds = gdal.Warp(
            clipped_saa_sza,
            resampled_saa_sza,
            cutlineWKT=polygon_wkt,
            cutlineSRS=raster_srs,
            cropToCutline=True,
            multithread=True,
            dstNodata=np.nan,
            outputType=gdal.GDT_Float32,
        )
    except RuntimeError:
        sza_ds = gdal.Warp(
            clipped_saa_sza,
            resampled_saa_sza,
            cutlineWKT=polygon_wkt,
            cutlineSRS="EPSG:4326",
            cropToCutline=True,
            multithread=True,
            dstNodata=np.nan,
            outputType=gdal.GDT_Float32,
        )
    # Resample and clip VAA_VZA file to match
    clipped_vaa_vza = "/vsimem/VAA_VZA_clipped.tif"
    try:
        vza_ds = gdal.Warp(
            clipped_vaa_vza,
            vaa_vza_file,
            cutlineWKT=polygon_wkt,
            cutlineSRS=raster_srs,
            cropToCutline=True,
            multithread=True,
            dstNodata=np.nan,
            outputType=gdal.GDT_Float32,
        )
    except RuntimeError:
        vza_ds = gdal.Warp(
            clipped_vaa_vza,
            vaa_vza_file,
            cutlineWKT=polygon_wkt,
            cutlineSRS="EPSG:4326",
            cropToCutline=True,
            multithread=True,
            dstNodata=np.nan,
            outputType=gdal.GDT_Float32,
        )

    # Extract SZA (Band 2 from SAA_SZA)
    gdal.Translate(sza_vrt, clipped_saa_sza, bandList=[2], format="VRT")

    # Extract VZA (Band 2 from VAA_VZA)
    gdal.Translate(vza_vrt, clipped_vaa_vza, bandList=[2], format="VRT")

    # Get actual size of clipped_vaa_vza
    clipped_ds = gdal.Open(clipped_vaa_vza)
    x_size = clipped_ds.RasterXSize
    y_size = clipped_ds.RasterYSize

    # Compute RAA (Relative Azimuth Angle) using a VRT pixel function
    raa_vrt_xml = f"""
    <VRTDataset rasterXSize="{x_size}" rasterYSize="{y_size}">
        <VRTRasterBand dataType="Float32" band="1">
            <PixelFunctionType>raa_pixel_function</PixelFunctionType>
            <SimpleSource>
                <SourceFilename relativeToVRT="1">{clipped_vaa_vza}</SourceFilename>
                <SourceBand>1</SourceBand>
            </SimpleSource>
            <SimpleSource>
                <SourceFilename relativeToVRT="1">{resampled_saa_sza}</SourceFilename>
                <SourceBand>1</SourceBand>
            </SimpleSource>
        </VRTRasterBand>
    </VRTDataset>
    """
    gdal.FileFromMemBuffer(raa_vrt, raa_vrt_xml.encode("utf-8"))

    return sza_ds, vza_ds, gdal.Open(raa_vrt)


def get_s2_siac_files(
    siac_folder: list[Path], start_date: str | dt.datetime, end_date: str | dt.datetime
):
    """
    Retrieves Sentinel-2 SIAC files within a specified date range from a given folder.

    Args:
        siac_folder (list[Path]): List of paths to the SIAC folders.
        start_date (str | dt.datetime): Start date of the range in 'YYYY-MM-DD' format or as a datetime object.
        end_date (str | dt.datetime): End date of the range in 'YYYY-MM-DD' format or as a datetime object.

    Returns:
        tuple: A tuple containing:
            - cloud_masks (dict): Dictionary with day of year (doy) as keys and paths to cloud mask files as values.
            - reflectances (dict): Dictionary with doy as keys and lists of paths to reflectance files as values.
            - angles (dict): Dictionary with doy as keys and tuples of paths to angle files (VZA and SAA) as values.
            - uncertainties (dict): Dictionary with doy as keys and lists of paths to uncertainty files as values.
            - doys (list): List of days of the year (doy) for which data is available.
    """
    if isinstance(start_date, str):
        start_date = dt.datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = dt.datetime.strptime(end_date, "%Y-%m-%d")
    file_list = scan_folder_siac(siac_folder)
    file_list = {
        #        ((k - start_date).days + 1): v
        k: v
        for k, v in file_list.items()
        if ((k >= start_date) and (k <= end_date))
    }
    # Sorty by DOY
    file_list = dict(sorted(file_list.items()))

    cloud_masks = {}
    reflectances = {}
    angles = {}
    uncertainties = {}
    doys = []

    for k, v in file_list.items():
        doy = int(k.strftime("%j"))
        if (v / "cloud.tif").exists():
            cloud_masks[doy] = v / "cloud.tif"
        else:
            continue
        refl_folder = v / "IMG_DATA"
        refl_files = [
            f for band in sel_bands for f in refl_folder.glob(f"*_*_{band}_sur.tif")
        ]
        refl_unc_files = [
            f for band in sel_bands for f in refl_folder.glob(f"*_*_{band}_sur_unc.tif")
        ]
        vza_file = v / "ANG_DATA/Mean_VAA_VZA.tif"
        saa_file = v / "ANG_DATA/SAA_SZA.tif"

        doys.append(doy)

        reflectances[doy] = refl_files
        uncertainties[doy] = refl_unc_files
        angles[doy] = (vza_file, saa_file)

    return cloud_masks, reflectances, angles, uncertainties, doys


def clip_raster(
    geojson_path: str | Path, raster: str | Path, resolution: float = 10.0
) -> np.ndarray:
    """Clip a raster file using a GeoJSON file.

    Parameters:
    geojson_path (str | Path): The path to the GeoJSON file used for clipping.
    raster (str | Path): The path to the raster file to be clipped.
    resolution (float): The resolution of the output raster.
    Returns:
    np.ndarray: The clipped raster as a pointer.

    Raises:
    IOError: If an error occurs while reading the raster file.
    """
    if isinstance(geojson_path, Path):
        geojson_path = str(geojson_path)
    if isinstance(raster, Path):
        raster = str(raster)
    raster_g = gdal.Warp(
        "",
        raster,
        format="MEM",
        cutlineDSName=geojson_path,
        cropToCutline=True,
        dstNodata=np.nan,
        outputType=gdal.GDT_Float32,
        xRes=resolution,
        yRes=resolution,
    )
    if raster_g is None:
        raise IOError("An error occurred while reading the file: {}".format(raster))
    return raster_g


def mask_cloud_snow(b2, b3, b4, b11):
    """Returns a binary mask where 1 = clouds/snow, 0 = clear"""
    ndsi = (b3 - b11) / (b3 + b11)

    bright = (b2 > 0.3) | (b3 > 0.3)  # High reflectance
    snow = (ndsi > 0.4) & bright
    clouds = (ndsi < 0.4) & bright & (b4 / b2 < 0.75)

    return (snow | clouds).astype(np.uint8)


def process_rasters(
    refl_files: dict[int, list[str]] | dict[int, list[Path]],
    refl_unc_files: dict[int, list[str]] | dict[int, list[Path]],
    cloud_files: dict[int, str] | dict[int, Path],
    angle_files: dict[int, tuple[str, str]] | dict[int, tuple[Path, Path]],
    geojson_path: str,
    min_sampling: int = None,
) -> tuple[np.ndarray, np.ndarray, list, np.ndarray, np.ndarray, tuple, str]:
    """
    Processes raster files by clipping them to a specified GeoJSON region and applying cloud masking.
    Args:
        refl_files (dict[int, list[str]] | dict[int, list[Path]]): Dictionary mapping day of year (DOY) to lists of reflectance file paths.
        refl_unc_files (dict[int, list[str]] | dict[int, list[Path]]): Dictionary mapping DOY to lists of reflectance uncertainty file paths.
        cloud_files (dict[int, str] | dict[int, Path]): Dictionary mapping DOY to cloud mask file paths.
        angle_files (dict[int, tuple[str, str]] | dict[int, tuple[Path, Path]]): Dictionary mapping DOY to tuples of angle file paths (solar and view angles).
        geojson_path (str): Path to the GeoJSON file defining the region of interest.
        min_sampling (int, optional): Minimum interval (in days) for DOY sampling. If set, ensures at least one observation every `min_sampling` days by adding NaN-filled placeholders.

    Returns:
        tuple: A tuple containing:
            - s2_reflectances (np.ndarray): Array of processed reflectance data.
            - s2_reflectances_unc (np.ndarray): Array of processed reflectance uncertainty data.
            - angles (np.ndarray): Array of median angles (solar zenith, view zenith, relative azimuth).
            - doys (list[int]): List of days of year corresponding to the processed data.
            - mask (np.ndarray): Boolean mask indicating where all reflectance data is NaN.
            - geotransform (tuple): Geotransform of the clipped raster.
            - crs (str): Coordinate reference system of the clipped raster.
    """
    s2_reflectances = []
    s2_reflectances_unc = []
    angles = []
    doys = []
    first_time = True
    for doy, cloud_file in cloud_files.items():
        cloud_data = clip_raster(geojson_path, cloud_file).ReadAsArray()
        refl_data = []
        refl_unc_data = []
        for refl_fname in refl_files[doy]:
            refl_data.append(clip_raster(geojson_path, refl_fname).ReadAsArray())
        for refl_fname_unc in refl_unc_files[doy]:
            refl_unc_data.append(
                clip_raster(geojson_path, refl_fname_unc).ReadAsArray()
            )
        refl_data = np.array(refl_data)

        cloud = cloud_data < 55  # I think...?
        # Now filter out snow
        cloud = cloud & (mask_cloud_snow(*refl_data[[0, 1, 2, 8]]) == 0)

        refl_data = np.where(cloud, refl_data / 10000.0, np.nan)
        refl_unc_data = np.ones_like(refl_data) * 0.01

        # fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
        # axs = axs.flatten()
        # axs[0].imshow(cloud_data, vmin=0, vmax=100)
        # im = axs[1].imshow(refl_data[7, :, :], vmin=0, vmax=0.8)
        # plt.colorbar(im, ax=axs[1])
        # fig.suptitle(f"DOY: {doy}")
        # plt.show()

        s2_reflectances.append(refl_data)
        refl_data_unc = np.where(cloud, np.nan, refl_unc_data / 10000.0)
        s2_reflectances_unc.append(refl_data_unc)

        sza, vza, raa = generate_angle_vrts(
            angle_files[doy][0], angle_files[doy][1], geojson_path
        )
        sza_data = np.nanmedian(sza.ReadAsArray() / 1000.0)
        vza_data = np.nanmedian(vza.ReadAsArray() / 1000.0)
        raa_data = np.nanmedian(raa.ReadAsArray() / 1000.0)

        angles.append([sza_data, vza_data, raa_data])
        doys.append(doy)
        if first_time:
            first_time = False
            g = clip_raster(geojson_path, refl_files[doy])
            geotransform = g.GetGeoTransform()
            crs = g.GetProjection()

    s2_reflectances = np.array(s2_reflectances)
    s2_reflectances_unc = np.array(s2_reflectances_unc)
    angles = list(np.array(angles).T)
    mask = np.all(np.isnan(s2_reflectances), axis=(0, 1))
    doys = np.array(doys)
    # If min_sampling is specified, ensure regular intervals
    if min_sampling is not None:
        expected_doys = np.arange(min(doys), max(doys) + 1, min_sampling)

        full_s2_reflectances = []
        full_s2_reflectances_unc = []
        full_angles = []
        full_doys = []
        full_masks = []

        for doy in expected_doys:
            if doy in doys:
                # Use existing data
                idx = np.where(doys == doy)[0][0]
                full_s2_reflectances.append(s2_reflectances[idx])
                full_s2_reflectances_unc.append(s2_reflectances_unc[idx])
                full_angles.append(angles[idx])
                full_masks.append(mask)
            else:
                # Insert NaN placeholders
                nan_layer = np.full_like(s2_reflectances[0], np.nan)
                full_s2_reflectances.append(nan_layer)
                full_s2_reflectances_unc.append(nan_layer)
                full_angles.append([np.nan, np.nan, np.nan])
                full_masks.append(np.ones_like(mask, dtype=bool))  # Full mask as True

            full_doys.append(doy)

        # Convert back to arrays
        s2_reflectances = np.array(full_s2_reflectances)
        s2_reflectances_unc = np.array(full_s2_reflectances_unc)
        angles = list(np.array(full_angles).T)
        mask = np.array(full_masks)
        doys = np.array(full_doys)

    return (s2_reflectances, s2_reflectances_unc, angles, doys, mask, geotransform, crs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt  # noqa

    siac_folder = "/home/jose/data/Bulgaria"
    start_date = "2016-01-15"
    end_date = "2016-07-25"
    cloud_masks, reflectances, angles, uncertainties, doys = get_s2_siac_files(
        siac_folder, start_date, end_date
    )
    json = "/mnt/me4084b/users/jose/Bulgaria/S2B_MSIL1C_20170819T091019_N0205_R050_T34TGP_20170819T091545.SAFE/GRANULE/L1C_T34TGP_A002364_20170819T091545/IMG_DATA/AOI.json"
    json = "/home/jose/python/ARC/bulgaria/field_1.geojson"
    (s2_reflectances, s2_reflectances_unc, angles, doys, mask, geotransform, crs) = (
        process_rasters(
            reflectances,
            uncertainties,
            cloud_masks,
            angles,
            json,  #
        )
    )
