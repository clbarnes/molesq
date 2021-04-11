import numpy as np


def is_iterable(obj):
    return hasattr(obj, "__iter__")


def grid_field(start, stop, samples=None, step=1):
    """Return coordinates of evenly-spaced samples in the given ROI,
    and the shape of an array of those coordinates.

    Sample spacing can be defined by the sample count or the step size.
    In each case, the value given can be a scalar,
    or a sequence the length of the dimensionality of start and stop.
    By default, there is a step of 1.

    The endpoint is not included.
    """
    ndim = len(start)
    if len(stop) != ndim:
        raise ValueError("start and stop must have same dimensionality")

    use_step = samples is None

    if not is_iterable(samples):
        samples = [samples] * ndim
    if not is_iterable(step):
        step = [step] * ndim

    if use_step:

        def fn(start_, stop_, _, step_):
            return np.arange(start_, stop_, step_)

    else:

        def fn(start_, stop_, samples_, _):
            return np.linspace(start_, stop_, samples_, endpoint=False)

    indices = [fn(*args) for args in zip(start, stop, samples, step)]
    shape = tuple(len(a) for a in indices)
    coords = np.stack(
        [m.reshape((-1,)) for m in np.meshgrid(*indices, indexing="ij")],
        axis=1,
    )
    return coords, shape
