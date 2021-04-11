import numpy as np

from molesq.utils import grid_field


def test_grid_field():
    test, shape = grid_field([0, 0], [2, 2])
    ref = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    ).astype(test.dtype)
    assert np.allclose(test, ref)
    assert shape == (2, 2)
