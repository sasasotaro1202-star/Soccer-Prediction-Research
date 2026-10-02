import numpy as np
from sklearn.linear_model import LogisticRegression

from src.research.target_permutation_adversarial import audit_target_permutation


def _fit_predict(X_train, y_train, X_eval, seed):
    model = LogisticRegression(max_iter=500, multi_class="auto", random_state=seed)
    model.fit(X_train, y_train)
    return model.predict_proba(X_eval)


def test_target_permutation_separates_real_signal_from_null():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(360, 5))
    y = np.select(
        [X[:, 0] + 0.4 * X[:, 1] > 0.8, X[:, 0] - 0.3 * X[:, 1] < -0.8],
        [0, 2],
        default=1,
    ).astype(int)
    report = audit_target_permutation(
        X_train=X[:250],
        y_train=y[:250],
        X_eval=X[250:],
        y_eval=y[250:],
        fit_predict=_fit_predict,
        seeds=(7, 19, 43, 71, 101, 137),
    )
    assert report["status"] == "SEPARATED"
    assert report["risk_flag"] is False
    assert report["selection_allowed"] is False


def test_target_permutation_requires_multiclass_labels():
    X = np.ones((40, 2))
    y = np.zeros(40, dtype=int)
    try:
        audit_target_permutation(
            X_train=X[:30], y_train=y[:30],
            X_eval=X[30:], y_eval=y[30:],
            fit_predict=_fit_predict,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected class validation failure")
