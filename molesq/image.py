from __future__ import annotations
from itertools import product

import numpy as np
from scipy.ndimage import map_coordinates

from .transform import Transformer
from .utils import grid_field


class ImageTransformer:
    def __init__(
        self,
        img,
        control_points,
        deformed_control_points,
        color_dim=None,
        interp_order=0,
        extrap_mode="constant",
        extrap_cval=0,
    ) -> None:
        self.img = img
        self.color_dim = color_dim
        if color_dim is None:
            self.channels = np.expand_dims(img, 0)
        else:
            self.channels = np.moveaxis(img, color_dim, 0)

        self.control_points = control_points
        self.deformed_control_points = deformed_control_points
        if self.control_points.shape[1] != self.channels.ndim - 1:
            raise ValueError("Dimensionality of image mismatches control points")

        self.interp_order = interp_order
        self.extrap_mode = extrap_mode
        self.extrap_cval = extrap_cval

        self.transformer = Transformer(
            self.control_points, self.deformed_control_points
        )
        self.img_shape = self.channels.shape[1:]

    def _map_coordinates(self, arr, coords):
        """coords as NxD, not scipy order of DxN"""
        return map_coordinates(
            arr,
            coords.T,
            order=self.interp_order,
            mode=self.extrap_mode,
            cval=self.extrap_cval,
        )

    def deform_viewport(self):
        """Deform the image, keeping the viewport the same.

        Some of the deformed image may be outside the field of view.
        """
        min_point = [0] * len(self.img_shape)
        max_point = self.img_shape
        return self.deform_arbitrary(min_point, max_point)

    def deform_whole(self):
        """Deform the entire image.

        The viewport is also translated and scaled
        to capture the entire extents of the deformed image.
        The translation of the viewport origin is also returned.
        """
        corners = (
            np.array(list(product([0, 1], repeat=len(self.img_shape)))) * self.img_shape
        )

        def_corners = self.transformer.transform(corners)
        min_point = np.min(def_corners, axis=0)
        max_point = np.max(def_corners, axis=0)
        return self.deform_arbitrary(min_point, max_point), min_point

    def deform_arbitrary(self, *args, **kwargs):
        """Deform by sampling a regular grid in deformed space.

        *args and **kwargs are passed to molesq.utils.grid_field.
        """
        eval_coords, coords_shape = grid_field(*args, **kwargs)
        coords = self.transformer.transform(eval_coords, True)

        if self.color_dim is None:
            return self._map_coordinates(self.img, coords).reshape(coords_shape)

        channels = [
            self._map_coordinates(c, coords).reshape(coords_shape)
            for c in self.channels
        ]

        return np.stack(channels, self.color_dim)
