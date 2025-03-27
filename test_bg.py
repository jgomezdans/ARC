from pathlib import Path
import numpy as np
import arc

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

the_fields = ["field_1.geojson", "field_2.geojson", "field_3.geojson",
              "field_4.geojson", "field_5.geojson", "field_6.geojson",
              ]
for year in [2023]:
    for field in the_fields:
        print(f"Doing {field}/{year}")
        start_date = f"{year}-10-01"
        end_date = f"{year+1}-10-01"
        geojson_path = f"/home/jose/python/ARC/bulgaria/{field}"
        geojson_path = Path(f"/home/jose/python/ARC/bulgaria/{field}")

        S2_data_folder = geojson_path.parent / geojson_path.stem / "S2_data"
        S2_data_folder.mkdir(parents=True, exist_ok=True)
        s2_refs, s2_uncs, s2_angles, doys, mask, geotransform, crs = arc.get_s2_official_data(
        start_date, end_date, geojson_path, S2_data_folder=S2_data_folder
        )
        ndvi = (s2_refs[:, 7] - s2_refs[:, 2]) / (s2_refs[:, 7] + s2_refs[:, 2])

        passer = np.nansum(ndvi, axis=(1,2)) > 0
        ndvi = ndvi[passer, :, :]
        doys = doys[passer]
        iloc = np.argmax(np.diff(doys, prepend=doys[0]) < 0)
        doys[iloc:] = doys[iloc:] + 365
        nrows = int(len(ndvi) / 5) + int(len(ndvi) % 5 > 0)

        fig, axs = plt.subplots(ncols=5, nrows=nrows, figsize=(10, 4 * nrows))
        axs = axs.ravel()
        cmap = plt.cm.YlGn
        cmap.set_bad("0.8")
        for i in range(len(axs)):
            if i < len(ndvi):

                im = axs[i].imshow(ndvi[i], vmin=0.2, vmax=1, cmap=cmap)
                #fig.colorbar(im, ax=axs[i])
                if doys[i] <= 365:
                    axs[i].set_title(f"DOY/{year}: {doys[i]}", fontsize=8)
                else:
                    axs[i].set_title(f"DOY/{year}: {doys[i] - 365}", fontsize=8)
                axs[i].set_xticks([])
                axs[i].set_yticks([])
            else:
                axs[i].set_visible(False)
        # Adjust space between subplots and for the colorbar
        fig.tight_layout()  # This reduces the space between subplots
        fig.subplots_adjust(right=0.85)  # Adjust the right space to fit the colorbar

        # Add the colorbar to the right of the subplots
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # Adjust these values for fine-tuning the colorbar
        fig.colorbar(im, cax=cbar_ax)
        fig.savefig(f"{field.split('.')[0]}_{year}.png", dpi=44)

        mean_ndvi = np.nanmean(ndvi, axis=(1, 2))
        plt.figure(figsize=(6, 4))
        mask = np.isfinite(mean_ndvi)
        plt.plot(doys[mask], mean_ndvi[mask], "o")
        plt.savefig(f"{field.split('.')[0]}_{year}_tseries.png", dpi=44)
        plt.close("all")
