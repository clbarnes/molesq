import sys

import numpy as np
from scipy.spatial.distance import cdist
from typing import Optional

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


def _transform_affine(locs, orig_cp, deformed_cp, cp_weights=None):
    """
    Makes heavy use of Einstein summation, resources here:

    * https://stackoverflow.com/questions/26089893/understanding-numpys-einsum
    * https://ajcr.net/Basic-guide-to-einsum/
    * Playground: https://oracleofnj.github.io/einsum-explainer/
    """
    n_locs, n_dim = locs.shape
    n_landmarks = len(orig_cp)

    # Pairwise distances between original control points and locations to transform
    # reshaped to 1,1,N_cp,N_l
    # jittered to avoid 0s

    if cp_weights is None:
        sqdists = cdist(orig_cp, locs, "sqeuclidean")
    else:
        # weights need to be factored in before squaring of distance, for unit reasons
        sqdists = (cdist(orig_cp, locs) / cp_weights[:, np.newaxis]) ** 2

    weights = 1 / (sqdists.reshape(1, 1, n_landmarks, n_locs) + sys.float_info.epsilon)

    weights_inverse_norm = 1 / np.sum(weights, axis=2)

    # reshape arrays for consistent indices
    orig_cp = reshape_points(orig_cp)
    deformed_cp = reshape_points(deformed_cp)

    # weighted centroids
    orig_star = np.einsum(
        "ijl,ijkl,ijkl->ijl", weights_inverse_norm, weights, orig_cp
    ).reshape(1, n_dim, 1, n_locs)
    deformed_star = np.einsum(
        "ijl,ijkl,ijkl->ijl", weights_inverse_norm, weights, deformed_cp
    ).reshape(1, n_dim, 1, n_locs)

    # distance to weighted centroids
    orig_hat = orig_cp - orig_star
    deformed_hat = deformed_cp - deformed_star

    Y = np.einsum("ijkl,mikl,mjkl->ijl", weights, orig_hat, orig_hat).reshape(
        n_dim, n_dim, 1, n_locs
    )

    rolled = np.moveaxis(Y, (0, 1), (2, 3))
    inv_rolled = np.linalg.inv(rolled)
    Y_inv = np.moveaxis(inv_rolled, (2, 3), (0, 1))

    Z = np.einsum("ijkl,mikl,mjkl->ijl", weights, orig_hat, deformed_hat).reshape(
        n_dim, n_dim, 1, n_locs
    )

    locs_reshaped = locs.ravel().reshape(1, n_dim, 1, n_locs, order=ORDER)
    vprime = (
        np.einsum("iakl,abkl,bjkl->ijkl", locs_reshaped - orig_star, Y_inv, Z)
        + deformed_star
    )
    vprime = vprime.ravel(ORDER).reshape(n_locs, n_dim)

    return vprime


class Transformer:
    def __init__(
        self,
        control_points: ArrayLike,
        deformed_control_points: ArrayLike,
        weights: Optional[ArrayLike] = None,
    ):
        """Class for transforming points using Moving Least Squares.

        Given control point arrays must both be of same shape NxD,
        where N is the number of points,
        and D the dimensionality.

        Parameters
        ----------
        control_points : ArrayLike
        deformed_control_points : ArrayLike
        weights : Optional[ArrayList]
            Any values <= 0 will be set to an arbitrarily small positive number

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
        self.control_points = from_arr
        self.deformed_control_points = to_arr

        if weights is not None:
            weights = np.asarray(weights)
            if weights.shape != (len(from_arr),):
                raise ValueError(
                    "weights must have same length as control points array"
                )
            weights[weights <= 0] = sys.float_info.epsilon

        self.weights = weights

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

        # if strategy == Strategy.AFFINE:
        #     return self._transform_affine(locs, orig_cp, deformed_cp)
        # else:
        #     raise ValueError(f"Unimplemented/ unknown strategy {strategy}")
        return _transform_affine(locs, orig_cp, deformed_cp, self.weights)
