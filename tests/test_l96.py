#test workflow 2
from Lorenz96.L96_multiscale import L96M

def test_l96_constructs():
    model = L96M()
    assert model is not None

def test_l96_dimension():
    model = L96M()
    assert model.K > 0
