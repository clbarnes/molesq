from __future__ import annotations
from itertools import product
from typing import Iterator
import sys

from .transform import Transformer

import numpy as np
from scipy.ndimage import map_coordinates

LESS_THAN_ONE = 1 - sys.float_info.epsilon


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
        self.control_points = control_points
        self.deformed_control_points = deformed_control_points
        self.color_dim = color_dim

        self.interp_order = interp_order
        self.extrap_mode = extrap_mode
        self.extrap_cval = extrap_cval

        self.transformer = Transformer(
            self.control_points, self.deformed_control_points
        )

        if color_dim is None:
            self.channel_shape = self.img.shape
            self.img_shape = self.img.shape
        else:
            img_shape = list(self.img.shape)
            img_shape[color_dim] = 1
            self.channel_shape = tuple(img_shape)
            img_shape.pop(color_dim)
            self.img_shape = tuple(img_shape)

    def with_deformed_control_points(self, deformed_control_points) -> ImageTransformer:
        return type(self)(
            self.img, self.control_points, deformed_control_points, self.color_dim
        )

    def _channels(self) -> Iterator[np.ndarray]:
        slices = [slice(None)] * self.img.ndim
        for idx in range(self.img.shape[self.color_dim]):
            slices[self.color_dim] = idx
            yield self.img[tuple(slices)]

    def _map_coordinates(self, arr, coords):
        return map_coordinates(
            arr,
            coords,
            order=self.interp_order,
            mode=self.extrap_mode,
            cval=self.extrap_cval,
        )

    def deform_whole_image(self):
        corners = np.array(
            list(product([0] * len(self.img_shape), np.array(self.img_shape)))
        )
        def_corners = self.transformer.transform(corners)
        min_point = np.min(def_corners, axis=0)
        max_point = np.max(def_corners, axis=0)
        grid_axes = [np.arange(mi, ma + 1) for mi, ma in zip(min_point, max_point)]
        channel_shape = [len(ga) for ga in grid_axes]
        def_coords = np.stack(
            [m.ravel() for m in np.meshgrid(*grid_axes, indexing="ij")],
            axis=1,
        )
        coords = self.transformer.transform(def_coords, True).T

        if self.color_dim is None:
            return self._map_coordinates(self.img, coords).reshape(channel_shape)

        channel_shape.insert(self.color_dim, 1)

        channels = [
            self._map_coordinates(c, coords).reshape(channel_shape)
            for c in self._channels()
        ]

        return np.concatenate(channels, self.color_dim)
