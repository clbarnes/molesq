import numpy as np

from molesq import Transformer
from .reference_impl import moving_least_squares_affine_vectorized


def test_importable():
    import molesq

    assert molesq.__version__


def test_instantiates(brain_landmarks):
    assert Transformer(*brain_landmarks)


def test_against_ref(brain_landmarks, neurons):
    n = neurons[0]
    t = Transformer(*brain_landmarks)
    test = t.transform(n)
    ref = moving_least_squares_affine_vectorized(n, *brain_landmarks)
    assert np.allclose(test, ref)


def test_ref_works(brain_landmarks, neurons):
    n = neurons[0]
    # t = Transformer(*brain_landmarks)
    # test = t.transform(n)
    ref = moving_least_squares_affine_vectorized(n, *brain_landmarks)
    assert n.shape == ref.shape
    assert not np.allclose(n, ref)
