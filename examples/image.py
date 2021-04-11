# %%
from pathlib import Path

import numpy as np
from imageio import imread
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from molesq import ImageTransformer
from molesq.utils import grid_field

DATA_DIR = Path(__file__).absolute().parent.parent / "data"

# %%

png = imread(DATA_DIR / "woody.png")[:, :, :-1]  # remove alpha channel
landmarks = np.genfromtxt(
    DATA_DIR / "woody_landmarks.tsv", delimiter="\t", usecols=(1, 2, 3, 4)
)
src = landmarks[:, :2]
tgt = landmarks[:, 2:]

# %%

trans = ImageTransformer(png, src, tgt, color_dim=2, interp_order=2)

field_coords, _ = grid_field([0, 0], np.array(trans.img_shape) + 1, 10)
deformed_field_coords = trans.transformer.transform(field_coords)

# %%

fig, ax_arr = plt.subplots(1, 3)
orig_ax: Axes = ax_arr[0]
viewport_ax: Axes = ax_arr[1]
whole_ax: Axes = ax_arr[2]

cmap = plt.get_cmap("Set1")

orig_ax.imshow(png)
orig_ax.scatter(*src.T[::-1], color=cmap.colors[: len(src)])
vu = deformed_field_coords - field_coords
orig_ax.quiver(
    field_coords[:, 1],
    field_coords[:, 0],
    vu[:, 1],
    vu[:, 0],
    scale=1.0,
    scale_units="xy",
    angles="xy",
)
orig_ax.set_title("\noriginal")

viewport = trans.deform_viewport()

viewport_ax.imshow(viewport)
viewport_ax.scatter(*tgt.T[::-1], color=cmap.colors[: len(tgt)])
viewport_ax.set_title("\ndeform_viewport")

whole, offset = trans.deform_whole()
whole_ax.imshow(whole)
whole_ax.scatter(*(tgt - offset).T[::-1], color=cmap.colors[: len(tgt)])
whole_ax.set_title(f"deform_whole\n(offset x{offset[1]:.1f} y{offset[0]:.1f})")

plt.show()
