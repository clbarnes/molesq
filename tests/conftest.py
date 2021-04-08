from pathlib import Path

import numpy as np

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


@pytest.fixture(scope="session")
def _neurons():
    return [read_swc(f) for f in sorted(DATA_DIR.glob("*.swc"))]


@pytest.fixture
def neurons(_neurons):
    return _neurons[0].copy(), _neurons[1].copy()


@pytest.fixture(scope="session")
def _brain_landmarks():
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
def brain_landmarks(_brain_landmarks):
    return _brain_landmarks[0].copy(), _brain_landmarks[1].copy()
