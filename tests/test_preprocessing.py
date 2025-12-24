import numpy as np
from src.data.preprocess import build_preprocessor

def test_preprocessor():
    prep = build_preprocessor()

    # simple dummy data
    X = np.array([[1, 2], [3, None]], dtype=object)

    # should run without crashing
    Xt = prep.fit_transform(X)

    # basic sanity checks
    assert Xt is not None
    assert Xt.shape[0] == 2
