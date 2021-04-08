import sys

import scipy as sp
import numpy as np


def moving_least_squares_affine_vectorized(vs, ps, qs):
    """
    Compute the 3d affine deformation based on Schaefer et al 2006.

    This implementation from
    https://github.com/ceesem/catalysis/blob/master/catalysis/transform.py

    Parameters
    ----------
    vs : Npoint x 3 numpy array
        Block of points to be transformed

    ps : Nlandmark x 3 numpy array
        Block of landmarks in the starting coordinates

    qs : Nlandmark x 3 numpy array
        Block of landmarks in the transformed coordinates, matched indices.

    Returns
    -------

    vprime : Npoint x 3 numpy array
        Block of points in the transformed coordinates.
    """
    Nlandmark = len(ps)
    Npoint = len(vs)

    # Reshape into arrays with consistent indices
    ps = ps.ravel().reshape(1, 3, Nlandmark, 1, order="F")
    qs = qs.ravel().reshape(1, 3, Nlandmark, 1, order="F")
    vs = vs.ravel().reshape(1, 3, 1, Npoint, order="F")

    ds = (
        sp.spatial.distance.cdist(
            ps.ravel("F").reshape(Nlandmark, 3),
            vs.ravel("F").reshape(Npoint, 3),
            "sqeuclidean",
        ).reshape(1, 1, Nlandmark, Npoint)
        + sys.float_info.epsilon
    )
    ws = 1 / ds

    wi_norm_inv = 1 / np.sum(ws, axis=2)
    pstar = np.einsum("ijl,ijkl,ijkl->ijl", wi_norm_inv, ws, ps).reshape(
        1, 3, 1, Npoint
    )
    qstar = np.einsum("ijl,ijkl,ijkl->ijl", wi_norm_inv, ws, qs).reshape(
        1, 3, 1, Npoint
    )

    phat = ps - pstar
    qhat = qs - qstar

    vminusp = vs - pstar

    Y = np.einsum("ijkl,mikl,mjkl->ijl", ws, phat, phat).reshape(3, 3, 1, Npoint)
    Yinv = np.zeros((3, 3, 1, Npoint))
    for i in range(Npoint):
        Yinv[:, :, 0, i] = np.linalg.inv(Y[:, :, 0, i])

    Z = np.einsum("ijkl,mikl,mjkl->ijl", ws, phat, qhat).reshape(3, 3, 1, Npoint)

    vprime = np.einsum("iakl,abkl,bjkl->ijkl", vminusp, Yinv, Z) + qstar
    vprime = vprime.ravel("F").reshape(Npoint, 3)
    return vprime
