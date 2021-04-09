from molesq import ImageTransformer


def test_image(woody, woody_landmarks):
    tran = ImageTransformer(
        woody,
        woody_landmarks[0],
        woody_landmarks[1],
        color_dim=2,
        extrap_cval=woody.max(),
    )
    deformed = tran.deform_whole_image()
    # same dimensionality
    assert deformed.ndim == woody.ndim
    # same color depth
    assert deformed.shape[-1] == woody.shape[-1]
    # non-empty
    assert deformed.max() > 0 and deformed.min() < 255
