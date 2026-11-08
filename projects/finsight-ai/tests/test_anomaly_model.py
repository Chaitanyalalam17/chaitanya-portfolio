
import numpy as np
from sklearn.ensemble import IsolationForest


def make_model():
    return IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )


def make_sample_data():
    rng = np.random.default_rng(42)
    return rng.normal(100, 20, size=(200, 2))


def test_model_training():
    model = make_model()
    data = make_sample_data()

    model.fit(data)

    assert hasattr(model, "estimators_")


def test_prediction_count():
    model = make_model()
    data = make_sample_data()

    predictions = model.fit_predict(data)

    assert len(predictions) == len(data)


def test_prediction_labels():
    model = make_model()
    data = make_sample_data()

    predictions = model.fit_predict(data)

    assert set(predictions).issubset({-1, 1})


def test_anomaly_scores():
    model = make_model()
    data = make_sample_data()

    model.fit(data)
    scores = model.decision_function(data)

    assert len(scores) == len(data)
    assert np.isfinite(scores).all()
