"""Our numpy inference must reproduce scikit-learn's predict_proba exactly.

This file uses scikit-learn directly, so the same relaxed rules as in _sklearn.py apply here.
"""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from bounded_distill._router import proba_from_logits
from bounded_distill._sklearn import fit_logistic_regression


@pytest.mark.parametrize("n_classes", [2, 3, 5])
def test_numpy_probabilities_match_sklearn(n_classes: int) -> None:
    rng = np.random.default_rng(n_classes)
    x = rng.normal(size=(300, 6)).astype(np.float32)
    y = rng.integers(n_classes, size=300).astype(np.intp)
    x[np.arange(300), y] += 2.0  # make classes learnable

    model = LogisticRegression(max_iter=1000).fit(x, y)
    weights, intercept = fit_logistic_regression(x, y)

    ours = proba_from_logits(x @ weights + intercept)
    # Router math runs in float64 (weights are float64). scikit-learn computes in the dtype of
    # its input, so it gets float64 input too; with float32 input it differs by ~1e-7.
    theirs = model.predict_proba(x.astype(np.float64))
    np.testing.assert_allclose(ours, theirs, rtol=1e-9, atol=1e-12)
