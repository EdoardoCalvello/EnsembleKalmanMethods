from Lorenz96.DATools_ import DATools

def test_datools_constructs():
    da = DATools()

    assert da is not None
    assert da.mem_thrsh == 500
    assert da.gpr is None

import numpy as np

from Lorenz96.DATools_ import DATools

def test_set_pairs():
    da = DATools()

    pairs = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    da.set_pairs(pairs)

    assert da.pairs is not None
    assert da.pairs.shape == (2, 2)