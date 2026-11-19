"""The only module that touches scikit-learn: a typed boundary around an untyped library.

scikit-learn ships no type stubs, so pyright strict reports every call as partially unknown.
Instead of scattering ignores across the code base, the relaxed rules below apply to this
file only, and every function converts results to precisely typed numpy arrays before
returning them.
"""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

_MAX_ITER = 1000


def cross_fit(
    x: npt.NDArray[np.float32],
    y: npt.NDArray[np.intp],
    n_classes: int,
    n_folds: int,
    random_state: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Fit one model per fold; return out-of-fold probabilities and every fold model.

    Returns ``(oof_proba, weights, intercepts)``. ``oof_proba[i]`` comes from the model that
    did not see row ``i``. The fold models are the router: averaging their probabilities on
    new texts gives confidences like the out-of-fold ones the threshold is chosen on, which
    a single model refit on all rows would not (it is more confident).

    ``weights[k]`` is fold ``k``'s ``coef_`` transposed, shape ``(dim, n_logits)``, so that
    ``logits = x @ weights[k] + intercepts[k]``. ``n_logits`` is the number of classes for
    three or more classes, but 1 for two classes (a single logit for class 1). ``y`` holds
    class indices ``0 .. n_classes - 1``; every class must have at least ``n_folds`` rows,
    so each training split contains every class and the columns line up with the indices.
    """
    proba = np.zeros((len(y), n_classes), dtype=np.float64)
    weights: list[npt.NDArray[np.float64]] = []
    intercepts: list[npt.NDArray[np.float64]] = []
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for train, held_out in folds.split(x, y):
        model = LogisticRegression(max_iter=_MAX_ITER)
        model.fit(x[train], y[train])
        proba[held_out] = np.asarray(model.predict_proba(x[held_out]), dtype=np.float64)
        weights.append(np.asarray(model.coef_.T, dtype=np.float64))
        intercepts.append(np.asarray(model.intercept_, dtype=np.float64))
    return proba, np.stack(weights), np.stack(intercepts)
