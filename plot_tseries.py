from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import random

# 0	N	1/100.
# 1	cab	1/100.
# 2	cm	1/10000.
# 3	cw	1/10000.
# 4	lai	1/100.
# 5	ala	1/100.
# 6	cbrown	1/1000.

variables = ["N", "Cab", "Cm", "Cw", "LAI", "ALA", "Cbrown"]
scalars = [
    1 / 100.0,
    1 / 100.0,
    1 / 10000.0,
    1 / 10000.0,
    1 / 100.0,
    1 / 100.0,
    1 / 1000.0,
]


loc = Path("/home/jose/python/ARC/bulgaria")
file_dumps = [f for f in loc.rglob("**/*.npz")]
for field_dump in file_dumps:
    f = np.load(field_dump)
    time_axis = [dt.datetime(2024, 1, 1) + dt.timedelta(days=int(i)) for i in f.f.doys]
    fig, axs = plt.subplots(nrows=2, ncols=4, sharex=True, figsize=(12, 8))
    axs = axs.flatten()

    colors = [
        "#FC8D62",
        "#66C2A5",
        "#8DA0CB",
        "#E78AC3",
        "#A6D854",
        "#FFD92F",
        "#E5C494",
        "#B3B3B3",
    ]
    random.shuffle(colors)
    for i, var in enumerate(variables):
        y = np.nanmean(f.f.post_bio_tensor, axis=0)[i, :] * scalars[i]
        y_std = np.nanstd(f.f.post_bio_tensor, axis=0)[i, :] * scalars[i]
        axs[i].plot(time_axis, y, "-o", color=colors[i])
        axs[i].fill_between(time_axis, y - y_std, y + y_std, fc=colors[i], alpha=0.5)
        axs[i].set_title(var)
        if i >= 3:
            axs[i].tick_params(labelrotation=45)

    axs[-1].set_visible(False)
    fig.suptitle(field_dump.stem)
    fig.tight_layout()
    fig.savefig(f"{field_dump.stem}_2024.pdf", dpi=144, bbox_inches="tight")
