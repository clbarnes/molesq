from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from molesq import Transformer

DATA_DIR = Path(__file__).absolute().parent.parent / "data"
SEED = 1991


def brain_landmarks():
    vals = np.genfromtxt(
        DATA_DIR / "Brain_Lineage_Landmarks_EMtoEM_ProjectSpace.csv",
        skip_header=1,
        usecols=(1, 2, 3, 4, 5, 6),
        delimiter=",",
    )
    left = vals[:, :3]
    right = vals[:, 3:]
    from_cp = np.vstack([left, right])
    to_cp = np.vstack([right, left])
    return from_cp, to_cp


def read_swc(fpath):
    return np.genfromtxt(
        fpath,
        usecols=(2, 3, 4),
        delimiter=" ",
    )


def morphologies():
    return [read_swc(f) for f in sorted(DATA_DIR.glob("*.swc"))]


def thin_coords(coords, discard=0.5):
    rng = np.random.default_rng(SEED)
    keep = rng.random(len(coords)) > discard
    return coords[keep]


def neurons():
    control_points, deformed_cp = brain_landmarks()

    # thin to improve performance of 3D renderer
    orig = thin_coords(np.concatenate(morphologies(), axis=0))

    tran = Transformer(control_points, deformed_cp)
    deformed = tran.transform(orig)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*orig.T, label="original neuron")
    ax.scatter(*deformed.T, label="deformed neuron")
    ax.scatter(*control_points.T, label="control points")
    # left/right mirror, so all control points are also deformed control points
    # ax.scatter(*deformed_cp.T, label="deformed CP")
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


if __name__ == "__main__":
    neurons()
