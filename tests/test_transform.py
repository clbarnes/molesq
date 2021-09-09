import numpy as np

import pytest
from molesq import Transformer
from .reference_impl import moving_least_squares_affine_vectorized


def test_importable():
    import molesq

    assert molesq.__version__


def test_instantiates(brain_landmarks):
    assert Transformer(*brain_landmarks)


def test_deforms_landmarks(brain_landmarks):
    src, tgt = brain_landmarks
    t = Transformer(src, tgt)
    deformed_src = t.transform(src)
    assert np.allclose(deformed_src, tgt)

    deformed_near_src = t.transform(np.nextafter(src, 1))
    assert np.allclose(deformed_near_src, tgt)


def test_reverse(brain_landmarks, neurons):
    n = neurons[0]
    t = Transformer(*brain_landmarks)
    deformed = t.transform(n)
    undeformed = t.transform(deformed, reverse=True)

    # check that no single undeformed point is too far away
    # from its original location,
    # defined as 0.5% of the diagonal of the neuron's axis aligned bounding box
    magnitude = np.linalg.norm(np.ptp(n, axis=1))
    assert np.max(np.abs(n - undeformed)) < magnitude * 0.005

    # TODO: difference between orig and undeformed > 2.4um for this neuron, seems bad?


def test_against_ref(brain_landmarks, neurons):
    n = neurons[0]
    t = Transformer(*brain_landmarks)
    test = t.transform(n)
    ref = moving_least_squares_affine_vectorized(n, *brain_landmarks)
    assert np.allclose(test, ref)


def test_ref_works(brain_landmarks, neurons):
    n = neurons[0]
    ref = moving_least_squares_affine_vectorized(n, *brain_landmarks)
    assert n.shape == ref.shape
    assert not np.allclose(n, ref)


def test_weighted(brain_landmarks, neurons):
    n = neurons[0]
    t = Transformer(*brain_landmarks)
    deformed = t.transform(n)

    rand = np.random.RandomState(1991)
    weights = rand.random(len(brain_landmarks[0]))
    t_weighted = Transformer(*brain_landmarks, weights=weights)
    deformed_weighted = t_weighted.transform(n)

    assert not np.allclose(n, deformed)
    assert not np.allclose(n, deformed_weighted)
    assert not np.allclose(deformed, deformed_weighted)


@pytest.mark.benchmark(group="points")
def test_benchmark(brain_landmarks, neurons, benchmark):
    n = neurons[0]
    t = Transformer(*brain_landmarks)
    benchmark(t.transform, n)


@pytest.mark.benchmark(group="points")
def test_benchmark_weighted(brain_landmarks, neurons, benchmark):
    n = neurons[0]
    rand = np.random.RandomState(1991)
    weights = rand.random(len(brain_landmarks[0]))
    t = Transformer(*brain_landmarks, weights=weights)
    benchmark(t.transform, n)


@pytest.mark.benchmark(group="points")
def test_benchmark_ref(brain_landmarks, neurons, benchmark):
    n = neurons[0]
    benchmark(moving_least_squares_affine_vectorized, n, *brain_landmarks)
