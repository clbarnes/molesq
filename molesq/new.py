import sys

import numpy as np
from scipy.spatial.distance import cdist

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum


class Strategy(StrEnum):
    AFFINE = "affine"
    # SIMILARITY = "similarity"
    # RIGID = "rigid"


ORDER = "F"


def reshape_landmarks(locations: np.ndarray) -> np.ndarray:
    """Reshape NxD landmark array into 1xDxNx1.

    Where D is the dimensionality, and N the number of landmarks.

    Parameters
    ----------
    locations

    Returns
    -------
    reshaped array
    """
    n_locations, n_dimensions = locations.shape
    return locations.ravel().reshape(1, n_dimensions, n_locations, 1, order=ORDER)


class Transformer:
    def __init__(self, from_landmarks, to_landmarks) -> None:
        from_arr = np.asarray(from_landmarks)
        to_arr = np.asarray(to_landmarks)

        if from_arr.shape != to_arr.shape:
            raise ValueError("Landmarks must have the same shape")
        if len(from_arr.shape) != 2:
            raise ValueError("Landmarks must be 2D array")

        self.n_landmarks, self.dimensions = from_arr.shape

        self.from_landmarks = reshape_landmarks(from_arr)
        self.to_landmarks = reshape_landmarks(to_arr)

    def _transform_affine(self, locs: np.ndarray, orig_cp, deformed_cp) -> np.ndarray:
        n_locs = locs.shape[-1]

        # Pairwise distances between original control points and locations to transform
        # reshaped to 1,1,N_cp,N_l
        # jittered to avoid 0s
        distances = (
            cdist(
                orig_cp.ravel(ORDER).reshape(self.n_landmarks, self.dimensions),
                locs.ravel(ORDER).reshape(n_locs, self.dimensions),
                "sqeuclidean",
            ).reshape(1, 1, self.n_landmarks, n_locs)
            + sys.float_info.epsilon
        )
        weights = 1 / distances

        weights_inverse_norm = 1 / np.sum(weights, axis=2)

        # weighted centroids
        orig_star = np.einsum(
            "ijl,ijkl,ijkl->ijl", weights_inverse_norm, weights, orig_cp
        ).reshape(1, self.dimensions, 1, n_locs)
        deformed_star = np.einsum(
            "ijl,ijkl,ijkl->ijl", weights_inverse_norm, weights, deformed_cp
        ).reshape(1, self.dimensions, 1, n_locs)

        # distance to weighted centroids
        orig_hat = orig_cp - orig_star
        deformed_hat = deformed_cp - deformed_star

        Y = np.einsum("ijkl,mikl,mjkl->ijl", weights, orig_hat, orig_hat).reshape(
            self.dimensions, self.dimensions, 1, n_locs
        )

        Y_inv = np.zeros((self.dimensions, self.dimensions, 1, n_locs))
        for i in range(n_locs):
            Y_inv[:, :, 0, i] = np.linalg.inv(Y[:, :, 0, i])

        Z = np.einsum("ijkl,mikl,mjkl->ijl", weights, orig_hat, deformed_hat).reshape(
            self.dimensions, self.dimensions, 1, n_locs
        )

        vprime = (
            np.einsum("iakl,abkl,bjkl->ijkl", locs - orig_star, Y_inv, Z)
            + deformed_star
        )
        vprime = vprime.ravel(ORDER).reshape(n_locs, self.dimensions)

        return vprime

    def transform(
        self,
        locations,
        inverse=False,
        strategy=Strategy.AFFINE,
    ) -> np.ndarray:
        locs = np.asarray(locations)

        orig_cp = self.from_landmarks
        deformed_cp = self.to_landmarks

        if inverse:
            orig_cp, deformed_cp = deformed_cp, orig_cp

        # Reshape into arrays with consistent indices
        locs = locs.ravel().reshape(1, self.dimensions, 1, len(locs), order=ORDER)

        if strategy == Strategy.AFFINE:
            return self._transform_affine(locs, orig_cp, deformed_cp)
        else:
            raise ValueError(f"Unimplemented/ unknown strategy {strategy}")
