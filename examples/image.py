#%%
from pathlib import Path

import numpy as np
from imageio import imread
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from molesq import ImageTransformer

DATA_DIR = Path(__file__).absolute().parent.parent / "data"

#%%

png = imread(DATA_DIR / "woody.png")[:, :, :-1]  # remove alpha channel
landmarks = np.genfromtxt(DATA_DIR / "woody_landmarks.tsv", delimiter="\t")
src = landmarks[:, :2]
tgt = landmarks[:, 2:]

#%%

trans = ImageTransformer(png, src, tgt, color_dim=2, extrap_cval=png.max())

deformed = trans.deform_whole_image()

# %%

fig, ax_arr = plt.subplots(1, 2)
orig_ax: Axes = ax_arr[0]
deformed_ax: Axes = ax_arr[1]

orig_ax.imshow(png)
deformed_ax.imshow(deformed)
plt.show()

# %%
