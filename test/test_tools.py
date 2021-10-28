import numpy as np
import pytest

from draco.util.random import _default_bitgen
from draco.util.tools import invert_no_zero

ARRAY_SIZE = (100, 111)
SEED = 12345
ATOL = 0.0
rng = np.random.Generator(_default_bitgen(SEED))

random_float_array = rng.standard_normal(size=ARRAY_SIZE, dtype=np.float32)
random_double_array = rng.standard_normal(size=ARRAY_SIZE, dtype=np.float64)
random_complex_array = rng.standard_normal(
    size=ARRAY_SIZE
) + 1.0j * rng.standard_normal(size=ARRAY_SIZE)


@pytest.mark.parametrize(
    "a", [random_complex_array, random_float_array, random_double_array]
)
def test_invert_no_zero(a):

    zero_ind = ((0, 10, 12), (56, 34, 78))
    good_ind = np.ones(a.shape, dtype=bool)
    good_ind[zero_ind] = False

    # set up some invalid values for inverse
    a[zero_ind[0][0], zero_ind[1][0]] = 0.0
    a[zero_ind[0][1], zero_ind[1][1]] = 0.5 / np.finfo(a.real.dtype).max

    if np.iscomplexobj(a):
        # these should be inverted fine
        a[10, 0] = 1.0
        a[10, 1] = 1.0j
        # also test invalid in the imaginary part
        a[zero_ind[0][2], zero_ind[1][2]] = 0.5j / np.finfo(a.real.dtype).max
    else:
        a[zero_ind[0][2], zero_ind[1][2]] = -0.5 / np.finfo(a.real.dtype).max

    b = invert_no_zero(a)
    assert np.allclose(b[good_ind], 1.0 / a[good_ind], atol=ATOL)
    assert (b[zero_ind] == 0).all()
