import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from src.data.preprocess import build_preprocessor

def test_preprocessor():
    prep = build_preprocessor()

    # simple dummy data as DataFrame (matching actual usage)
    X = pd.DataFrame({
        'feature1': [1, 3],
        'feature2': [2, np.nan]
    })

    # should run without crashing
    Xt = prep.fit_transform(X)

    # basic sanity checks
    assert Xt is not None
    assert Xt.shape[0] == 2
    assert Xt.shape[1] == 2  # Should have 2 features
