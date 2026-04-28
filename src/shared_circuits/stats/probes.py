"""Logistic-regression probes over residual-stream activations."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


def train_probe(
    x_pos: np.ndarray,
    x_neg: np.ndarray,
    max_iter: int = 2000,
    c: float = 1.0,
) -> dict[str, float | np.ndarray]:
    """
    Train a logistic regression probe on positive vs negative activations.

    Args:
        x_pos: Activations for positive class (e.g. wrong-opinion).
        x_neg: Activations for negative class (e.g. correct-opinion).
        max_iter: Maximum iterations for solver.
        c: Regularization strength (inverse).

    Returns:
        Dict with auroc, accuracy, coefficients, and intercept.

    """
    x = np.concatenate([x_pos, x_neg])
    y = np.array([1] * len(x_pos) + [0] * len(x_neg))
    clf = LogisticRegression(max_iter=max_iter, C=c)
    clf.fit(x, y)
    proba = clf.predict_proba(x)[:, 1]
    preds = clf.predict(x)
    return {
        'auroc': float(roc_auc_score(y, proba)),
        'accuracy': float(accuracy_score(y, preds)),
        'coefficients': clf.coef_[0],
        'intercept': float(clf.intercept_[0]),
    }


def evaluate_probe_transfer(
    x_train_pos: np.ndarray,
    x_train_neg: np.ndarray,
    x_test_pos: np.ndarray,
    x_test_neg: np.ndarray,
    max_iter: int = 2000,
    c: float = 1.0,
) -> dict[str, float]:
    """
    Train a probe on one task, evaluate on another (transfer test).

    Args:
        x_train_pos: Training positive activations.
        x_train_neg: Training negative activations.
        x_test_pos: Test positive activations.
        x_test_neg: Test negative activations.
        max_iter: Maximum solver iterations.
        c: Regularization strength.

    Returns:
        Dict with train_auroc, test_auroc, train_accuracy, test_accuracy.

    """
    x_train = np.concatenate([x_train_pos, x_train_neg])
    y_train = np.array([1] * len(x_train_pos) + [0] * len(x_train_neg))
    x_test = np.concatenate([x_test_pos, x_test_neg])
    y_test = np.array([1] * len(x_test_pos) + [0] * len(x_test_neg))

    clf = LogisticRegression(max_iter=max_iter, C=c)
    clf.fit(x_train, y_train)

    train_proba = clf.predict_proba(x_train)[:, 1]
    test_proba = clf.predict_proba(x_test)[:, 1]

    return {
        'train_auroc': float(roc_auc_score(y_train, train_proba)),
        'test_auroc': float(roc_auc_score(y_test, test_proba)),
        'train_accuracy': float(accuracy_score(y_train, clf.predict(x_train))),
        'test_accuracy': float(accuracy_score(y_test, clf.predict(x_test))),
    }
