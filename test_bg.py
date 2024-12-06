from pathlib import Path
import numpy as np
import arc

import matplotlib.pyplot as plt


from typing import List, Union
from scipy.optimize import least_squares


def double_logistic(
    p: List[float], t: Union[float, np.ndarray]
) -> np.ndarray:
    """
    Computes the double sigmoid logistic function.

    Parameters:
        p (List[float]): List of parameters for the logistic function.
        t (Union[float, np.ndarray]): Input values.

    Returns:
        np.ndarray: Output values computed using the logistic function.
    """
    assert (
        len(p) == 6
    ), "The parameter list p should contain exactly six elements."

    sigma1 = 1.0 / (1 + np.exp(p[2] * (t - p[3])))
    sigma2 = 1.0 / (1 + np.exp(-p[4] * (t - p[5])))

    return p[0] - p[1] * (sigma1 + sigma2 - 1)


def compute_difference(
    p: List[float], x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """
    Computes the difference between a logistic function and provided y values.

    Parameters:
        p (List[float]): List of parameters for the logistic function.
        x (np.ndarray): Input values for the logistic function.
        y (np.ndarray): Target y values to compare with the logistic function's results.

    Returns:
        np.ndarray: Difference between the logistic function and y values. Any non-finite values are replaced with 0.
    """
    # Start has to be before end, and the difference between them has to be at least 10 days
    if (p[3] >= p[5]) and (p[5] - p[3]) < 10:
        return np.ones_like(y) * 2
    # Compute logistic function with given parameters and inputs
    result = double_logistic(p, x)

    # Calculate difference between logistic function results and target y values
    difference = result - y

    # Replace any non-finite values in the difference array with 0
    difference[~np.isfinite(difference)] = 0

    return difference


def fit_dbl_logistic(
    doys: np.ndarray,
    mean_ndvi: np.ndarray,
    init_vals: np.ndarray | None = None,
) -> np.ndarray:
    NUM_DAYS = 365
    INITIAL_PARAMETERS = (
        0.2,
        0.7,
        0.1,
        None,
        0.1,
        None,
    )  # Placeholder values for the start and end days
    BOUNDS = np.array(
        [[0, 1], [0, 0.95], [0, 1], [0, NUM_DAYS], [0, 1], [0, NUM_DAYS]]
    ).T
    doys = doys - doys[0]
    mid_index = doys[np.argsort(mean_ndvi)[::-1][0]]

    start_day = max([min([(mid_index + 0) / 2, NUM_DAYS]), 0])
    end_day = max([min([(mid_index + NUM_DAYS) / 2, NUM_DAYS]), 0])

    initial_parameters = list(INITIAL_PARAMETERS)
    initial_parameters[3] = start_day
    initial_parameters[5] = end_day
    if init_vals is not None:
        # Use the average of the initial values from a different site
        # and the initial parameters
        initial_parameters = 0.5 * (
            np.array(initial_parameters) + np.array(init_vals)
        )
    result = least_squares(
        compute_difference,
        initial_parameters,
        loss="soft_l1",
        f_scale=0.001,
        args=(doys, mean_ndvi),
        bounds=BOUNDS,
    )
    print(result)
    return tuple(result.x)


the_fields = [
    "field_1.geojson",
    "field_2.geojson",
    "field_3.geojson",
    "field_4.geojson",
    "field_5.geojson",
    "field_6.geojson",
]
fitted_params = {}
for year in [2019, 2020, 2021, 2022, 2023, 2024]:
    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(12, 8))
    axs = axs.ravel()
    p = None
    fitted_params[year] = []
    for ii, field in enumerate(the_fields):
        print(f"Doing {field}/{year}")
        start_date = f"{year}-01-01"
        end_date = f"{year}-10-15"
        geojson_path = f"/home/jose/python/ARC/bulgaria/{field}"
        geojson_path = Path(f"/home/jose/python/ARC/bulgaria/{field}")

        S2_data_folder = geojson_path.parent / geojson_path.stem / "S2_data"
        S2_data_folder.mkdir(parents=True, exist_ok=True)
        s2_refs, s2_uncs, s2_angles, doys, mask, geotransform, crs = (
            arc.get_s2_official_data(
                start_date,
                end_date,
                geojson_path,
                S2_data_folder=S2_data_folder,
            )
        )
        ndvi = (s2_refs[:, 7] - s2_refs[:, 2]) / (
            s2_refs[:, 7] + s2_refs[:, 2]
        )

        passer = np.nansum(ndvi, axis=(1, 2)) > 0
        ndvi = ndvi[passer, :, :]
        doys = doys[passer]
        # iloc = np.argmax(np.diff(doys, prepend=doys[0]) < 0)
        # doys[iloc:] = doys[iloc:] + 365
        nrows = int(len(ndvi) / 5) + int(len(ndvi) % 5 > 0)

        # fig, axs = plt.subplots(ncols=5, nrows=nrows, figsize=(10, 4 * nrows))
        # axs = axs.ravel()
        # cmap = plt.cm.YlGn
        # cmap.set_bad("0.8")
        # for i in range(len(axs)):
        #     if i < len(ndvi):
        #         im = axs[i].imshow(ndvi[i], vmin=0.2, vmax=1, cmap=cmap)
        #         # fig.colorbar(im, ax=axs[i])
        #         if doys[i] <= 365:
        #             axs[i].set_title(f"DOY/{year}: {doys[i]}", fontsize=8)
        #         else:
        #             axs[i].set_title(
        #                 f"DOY/{year}: {doys[i] - 365}", fontsize=8
        #             )
        #         axs[i].set_xticks([])
        #         axs[i].set_yticks([])
        #     else:
        #         axs[i].set_visible(False)
        # # Adjust space between subplots and for the colorbar
        # fig.tight_layout()  # This reduces the space between subplots
        # fig.subplots_adjust(
        #     right=0.85
        # )  # Adjust the right space to fit the colorbar

        # # Add the colorbar to the right of the subplots
        # cbar_ax = fig.add_axes(
        #     [0.88, 0.15, 0.02, 0.7]
        # )  # Adjust these values for fine-tuning the colorbar
        # fig.colorbar(im, cax=cbar_ax)
        # fig.savefig(f"{field.split('.')[0]}_{year}.png", dpi=44)

        mean_ndvi = np.nanmean(ndvi, axis=(1, 2))
        p = None
        if p is None:
            p = fit_dbl_logistic(doys, mean_ndvi)
        else:
            p = fit_dbl_logistic(doys, mean_ndvi, init_vals=p)

        print(p)
        fitted_params[year].append(p)
        # plt.figure(figsize=(6, 4))
        mask = np.isfinite(mean_ndvi)
        axs[ii].plot(doys[mask], mean_ndvi[mask], "o", label="Mean NDVI")
        axs[ii].plot(
            doys,
            double_logistic(p, doys - doys[0]),
            "--",
            label="Fitted double logistic",
        )
        if ii == 0:
            axs[ii].legend(loc="best", frameon=False, fontsize=8)
        axs[ii].set_title(
            f"{field.split('.')[0]} VI_max: {p[0] + p[1]:.3f}\n SOS: {p[3]:.1f}, EOS: {p[5]:.1f}",
            fontsize=9,
        )
        axs[ii].set_xlabel("DOY")
        axs[ii].set_ylabel("Mean NDVI")
    fig.tight_layout()
    fig.savefig(f"allfields_{year}_tseries.pdf", dpi=300, bbox_inches="tight")
    plt.close("all")
