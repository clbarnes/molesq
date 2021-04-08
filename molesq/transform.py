import sys

import numpy as np
from scipy.spatial.distance import cdist

from numpy.typing import ArrayLike

# try:
#     from enum import StrEnum
# except ImportError:
#     from backports.strenum import StrEnum


# class Strategy(StrEnum):
#     AFFINE = "affine"
#     # SIMILARITY = "similarity"
#     # RIGID = "rigid"


ORDER = "F"


def reshape_points(control_points: np.ndarray) -> np.ndarray:
    """Reshape NxD array into 1xDxNx1.

    Where D is the dimensionality, and N the number of points.

    Parameters
    ----------
    locations

    Returns
    -------
    reshaped array
    """
    n_locations, n_dimensions = control_points.shape
    return control_points.ravel().reshape(1, n_dimensions, n_locations, 1, order=ORDER)


class Transformer:
    def __init__(self, control_points: ArrayLike, deformed_control_points: ArrayLike):
        """Class for transforming points using Moving Least Squares.

        Given control point arrays must both be of same shape NxD,
        where N is the number of points,
        and D the dimensionality.

        Parameters
        ----------
        control_points : ArrayLike
        deformed_control_points : ArrayLike

        Raises
        ------
        ValueError
            Invalid control point array(s)
        """
        from_arr = np.asarray(control_points)
        to_arr = np.asarray(deformed_control_points)

        if from_arr.shape != to_arr.shape:
            raise ValueError("Control points must have the same shape")
        if from_arr.ndim != 2:
            raise ValueError("Control points must be 2D array")

        self.n_landmarks, self.ndim = from_arr.shape

        self.control_points: np.ndarray = reshape_points(from_arr)
        self.deformed_control_points: np.ndarray = reshape_points(to_arr)

    def _transform_affine(
        self, locs: np.ndarray, orig_cp: np.ndarray, deformed_cp: np.ndarray
    ) -> np.ndarray:
        n_locs = locs.shape[-1]

        # Pairwise distances between original control points and locations to transform
        # reshaped to 1,1,N_cp,N_l
        # jittered to avoid 0s
        distances = (
            cdist(
                orig_cp.ravel(ORDER).reshape(self.n_landmarks, self.ndim),
                locs.ravel(ORDER).reshape(n_locs, self.ndim),
                "sqeuclidean",
            ).reshape(1, 1, self.n_landmarks, n_locs)
            + sys.float_info.epsilon
        )
        weights = 1 / distances

        weights_inverse_norm = 1 / np.sum(weights, axis=2)

        # weighted centroids
        orig_star = np.einsum(
            "ijl,ijkl,ijkl->ijl", weights_inverse_norm, weights, orig_cp
        ).reshape(1, self.ndim, 1, n_locs)
        deformed_star = np.einsum(
            "ijl,ijkl,ijkl->ijl", weights_inverse_norm, weights, deformed_cp
        ).reshape(1, self.ndim, 1, n_locs)

        # distance to weighted centroids
        orig_hat = orig_cp - orig_star
        deformed_hat = deformed_cp - deformed_star

        Y = np.einsum("ijkl,mikl,mjkl->ijl", weights, orig_hat, orig_hat).reshape(
            self.ndim, self.ndim, 1, n_locs
        )

        Y_inv = np.zeros((self.ndim, self.ndim, 1, n_locs))
        for i in range(n_locs):
            Y_inv[:, :, 0, i] = np.linalg.inv(Y[:, :, 0, i])

        Z = np.einsum("ijkl,mikl,mjkl->ijl", weights, orig_hat, deformed_hat).reshape(
            self.ndim, self.ndim, 1, n_locs
        )

        vprime = (
            np.einsum("iakl,abkl,bjkl->ijkl", locs - orig_star, Y_inv, Z)
            + deformed_star
        )
        vprime = vprime.ravel(ORDER).reshape(n_locs, self.ndim)

        return vprime

    def transform(
        self,
        locations: ArrayLike,
        reverse=False,
        # strategy=Strategy.AFFINE,
    ) -> np.ndarray:
        """Transform some locations using the given control points.

        Uses the affine form of the MLS algorithm.

        Parameters
        ----------
        locations : ArrayLike
            NxD array of N locations in D dimensions to transform.
        reverse : bool, optional
            Transform from deformed space to original space, by default False

        Returns
        -------
        Deformed points
        """
        locs = np.asarray(locations)
        if locs.ndim != 2 or locs.shape[-1] != self.ndim:
            raise ValueError(
                "Locations must be 2D array of same width as control points, "
                f"got {locs.shape}"
            )

        orig_cp = self.control_points
        deformed_cp = self.deformed_control_points

        if reverse:
            orig_cp, deformed_cp = deformed_cp, orig_cp

        # Reshape into arrays with consistent indices
        locs = locs.ravel().reshape(1, self.ndim, 1, len(locs), order=ORDER)

        # if strategy == Strategy.AFFINE:
        #     return self._transform_affine(locs, orig_cp, deformed_cp)
        # else:
        #     raise ValueError(f"Unimplemented/ unknown strategy {strategy}")
        return self._transform_affine(locs, orig_cp, deformed_cp)
