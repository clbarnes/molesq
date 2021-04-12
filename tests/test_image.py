from molesq import ImageTransformer


def test_deform_viewport(woody, woody_landmarks):
    tran = ImageTransformer(
        woody,
        woody_landmarks[0],
        woody_landmarks[1],
        color_dim=2,
        extrap_mode="nearest",
    )
    deformed = tran.deform_viewport()
    # same dimensionality
    assert deformed.shape == woody.shape
    # non-empty
    assert deformed.max() > 0 and deformed.min() < 255


def test_bench_viewport(woody, woody_landmarks, benchmark):
    tran = ImageTransformer(
        woody,
        woody_landmarks[0],
        woody_landmarks[1],
        color_dim=2,
        extrap_mode="nearest",
    )
    benchmark(tran.deform_viewport)
