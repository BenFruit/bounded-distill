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


def out_of_fold_proba(
    x: npt.NDArray[np.float32],
    y: npt.NDArray[np.intp],
    n_classes: int,
    n_folds: int,
    random_state: int,
) -> npt.NDArray[np.float64]:
    """Class probabilities for every row, each from a model that did not see that row.

    ``y`` holds class indices ``0 .. n_classes - 1``; every class must have at least
    ``n_folds`` rows, so each training split contains every class and the probability
    columns always line up with the class indices.
    """
    proba = np.zeros((len(y), n_classes), dtype=np.float64)
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for train, held_out in folds.split(x, y):
        model = LogisticRegression(max_iter=_MAX_ITER)
        model.fit(x[train], y[train])
        proba[held_out] = np.asarray(model.predict_proba(x[held_out]), dtype=np.float64)
    return proba


def fit_logistic_regression(
    x: npt.NDArray[np.float32], y: npt.NDArray[np.intp]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Fit on all rows and return ``(weights, intercept)`` so inference needs only numpy.

    ``logits = x @ weights + intercept``. ``weights`` is scikit-learn's ``coef_`` transposed,
    shape ``(dim, n_logits)``, so prediction needs no transpose. ``n_logits`` is the number
    of classes for three or more classes, but 1 for two classes (a single logit for class 1).
    """
    model = LogisticRegression(max_iter=_MAX_ITER)
    model.fit(x, y)
    weights = np.asarray(model.coef_.T, dtype=np.float64)
    intercept = np.asarray(model.intercept_, dtype=np.float64)
    return weights, intercept
