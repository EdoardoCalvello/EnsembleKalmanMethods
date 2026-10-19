from joblib import load

def test_gpr_loads():
    gp = load("Lorenz96/closure.joblib")
    assert gp is not None