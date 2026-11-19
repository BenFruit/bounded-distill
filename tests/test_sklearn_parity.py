"""Our numpy inference must reproduce scikit-learn's predict_proba for every fold model.

This file uses scikit-learn directly, so the same relaxed rules as in _sklearn.py apply here.
"""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from bounded_distill._router import proba_from_logits
from bounded_distill._sklearn import cross_fit


@pytest.mark.parametrize("n_classes", [2, 3, 5])
def test_fold_models_reproduce_out_of_fold_probabilities(n_classes: int) -> None:
    rng = np.random.default_rng(n_classes)
    x = rng.normal(size=(300, 6)).astype(np.float32)
    y = rng.integers(n_classes, size=300).astype(np.intp)
    x[np.arange(300), y] += 2.0  # make classes learnable

    oof, weights, intercepts = cross_fit(x, y, n_classes, n_folds=4, random_state=0)

    n_logits = 1 if n_classes == 2 else n_classes
    assert weights.shape == (4, 6, n_logits)
    assert intercepts.shape == (4, n_logits)
    folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=0).split(x, y)
    for k, (_, held_out) in enumerate(folds):
        ours = proba_from_logits(x[held_out] @ weights[k] + intercepts[k])
        # Router math runs in float64; scikit-learn computed the out-of-fold values in the
        # float32 of its input, which differs by ~1e-7.
        np.testing.assert_allclose(ours, oof[held_out], atol=1e-6)
