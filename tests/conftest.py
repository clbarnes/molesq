from pathlib import Path

import numpy as np
from imageio import imread

import pytest

TEST_DIR = Path(__file__).absolute().parent
PROJECT_DIR = TEST_DIR.parent
DATA_DIR = PROJECT_DIR / "data"


def read_swc(fpath):
    return np.genfromtxt(
        fpath,
        usecols=(2, 3, 4),
        delimiter=" ",
    )


@pytest.fixture
def neurons():
    return [read_swc(f) for f in sorted(DATA_DIR.glob("*.swc"))]


@pytest.fixture
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


@pytest.fixture
def woody():
    return imread(DATA_DIR / "woody.png")[:, :, :-1]


@pytest.fixture
def woody_landmarks():
    vals = np.genfromtxt(
        DATA_DIR / "woody_landmarks.tsv", delimiter="\t", usecols=(1, 2, 3, 4)
    )
    return vals[:, :2], vals[:, 2:]
