"""FastAPI serving layer for the spike forecast and the dispatch engine.

``POST /api/v1/grid/forecast`` scores the saved LightGBM model and returns the
top three probability-scale SHAP attributions. ``POST /api/v1/grid/simulate``
does not call the model. It applies the caller-supplied spike probabilities
through the deterministic dispatch rules. A missing model file does not block
simulation, and model output never authorizes a breaker by itself.
"""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from src.api.schemas import (
    SPIKE_FEATURE_COLUMNS,
    DispatchSimulationRequest,
    ExplainResponse,
    FeatureAttribution,
    FeatureContribution,
    ForecastRequest,
    GridSimulationResponse,
    PredictResponse,
    ShapAttributionPayload,
    SpikeFeatureVector,
)
from src.models.dispatch_optimizer import simulate_dispatch

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "models" / "spike_lgb_v1.pkl"
EXPLAINER_PATH = REPO_ROOT / "models" / "explainer.pkl"
MODEL_NAME = "spike_lgb_v1"

app = FastAPI(
    title="ERCOT Houston Hub Forecast and Dispatch",
    version="0.2.0",
    summary="Read-only spike probability and deterministic curtailment economics.",
)

_lock = Lock()
_artifacts: dict[str, object] = {}


@app.get("/")
def root() -> dict[str, object]:
    """Service metadata and the routes a client should call next."""

    return {
        "service": app.title,
        "version": app.version,
        "model_name": MODEL_NAME,
        "links": {
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
            "health": "/health",
            "predict": "/predict",
            "explain": "/explain",
            "forecast": "/api/v1/grid/forecast",
            "simulate": "/api/v1/grid/simulate",
        },
    }


@app.get("/health")
def health() -> dict[str, object]:
    """Report process liveness and whether the spike artifacts are on disk."""

    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "model_loaded": MODEL_PATH.is_file() and EXPLAINER_PATH.is_file(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(
    features: SpikeFeatureVector,
    decision_threshold: float = Query(default=0.5, gt=0.0, lt=1.0),
) -> PredictResponse:
    """Score one feature dictionary and return P(spike) plus the alert flag."""

    artifact, probability, _base_value, _ranked = _score(features)
    return PredictResponse(
        model_name=str(artifact.get("model_name", MODEL_NAME)),
        probability_spike_250=probability,
        spike_alert=probability >= decision_threshold,
        decision_threshold=decision_threshold,
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(features: SpikeFeatureVector) -> ExplainResponse:
    """Return every TreeExplainer contribution, largest absolute value first."""

    artifact, probability, base_value, ranked = _score(features)
    contributions = [
        FeatureContribution(
            rank=index,
            feature_name=name,
            feature_value=feature_value,
            shap_value=shap_value,
            abs_shap_value=abs(shap_value),
        )
        for index, (name, feature_value, shap_value) in enumerate(ranked, start=1)
    ]
    return ExplainResponse(
        model_name=str(artifact.get("model_name", MODEL_NAME)),
        probability_spike_250=probability,
        base_value=base_value,
        contributions=contributions,
    )


@app.post("/api/v1/grid/forecast", response_model=ShapAttributionPayload)
def forecast(request: ForecastRequest) -> ShapAttributionPayload:
    """Return P(LMP > $250/MWh) and the three largest SHAP attributions."""

    artifact, probability, base_value, ranked = _score(request.features)
    top = [
        FeatureAttribution(
            rank=index,
            feature_name=name,
            feature_value=feature_value,
            shap_value=shap_value,
            abs_shap_value=abs(shap_value),
        )
        for index, (name, feature_value, shap_value) in enumerate(ranked[:3], start=1)
    ]
    return ShapAttributionPayload(
        model_name=str(artifact.get("model_name", MODEL_NAME)),
        model_version=str(artifact.get("model_version", "1")),
        as_of_utc=request.as_of_utc,
        horizon_minutes=request.horizon_minutes,
        probability_spike_250=probability,
        predicted_positive=probability >= request.decision_threshold,
        decision_threshold=request.decision_threshold,
        base_value=base_value,
        top_features=top,
    )


@app.post("/api/v1/grid/simulate", response_model=GridSimulationResponse)
def simulate(request: DispatchSimulationRequest) -> GridSimulationResponse:
    """Compare baseline, curtailment, and battery-plus-curtailment on a price path."""

    try:
        return simulate_dispatch(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _score(
    features: SpikeFeatureVector,
) -> tuple[dict[str, object], float, float, list[tuple[str, float, float]]]:
    """Run predict_proba and the saved TreeExplainer on one feature row."""

    artifact, explainer = _load_artifacts()
    frame = pd.DataFrame(
        [{name: getattr(features, name) for name in SPIKE_FEATURE_COLUMNS}],
        columns=list(SPIKE_FEATURE_COLUMNS),
    )
    model = artifact["model"]
    probability = float(model.predict_proba(frame)[0, 1])
    probability = min(1.0, max(0.0, probability))
    explanation = explainer(frame)
    values = np.asarray(explanation.values).reshape(-1)
    base_value = float(np.asarray(explanation.base_values).reshape(-1)[0])
    if len(values) != len(SPIKE_FEATURE_COLUMNS):
        raise HTTPException(status_code=500, detail="SHAP returned a feature vector that does not match the model.")
    ranked = sorted(
        (
            (name, float(frame.iloc[0][name]), float(value))
            for name, value in zip(SPIKE_FEATURE_COLUMNS, values, strict=True)
        ),
        key=lambda item: (-abs(item[2]), item[0]),
    )
    return artifact, probability, base_value, ranked


def _load_artifacts() -> tuple[dict[str, object], object]:
    if _artifacts:
        return _artifacts["bundle"], _artifacts["explainer"]  # type: ignore[return-value]
    if not MODEL_PATH.is_file() or not EXPLAINER_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="Spike model artifacts are missing. Run python -m src.models.train_classifier.",
        )
    with _lock:
        if _artifacts:
            return _artifacts["bundle"], _artifacts["explainer"]  # type: ignore[return-value]
        try:
            import joblib
            import lightgbm  # noqa: F401
            import shap  # noqa: F401

            bundle = joblib.load(MODEL_PATH)
            explainer = joblib.load(EXPLAINER_PATH)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Spike model artifacts could not be loaded ({exc.__class__.__name__}).") from exc
        columns = list(bundle.get("feature_columns", []))
        if columns != list(SPIKE_FEATURE_COLUMNS):
            raise HTTPException(status_code=503, detail="Saved model features do not match the serving contract.")
        _artifacts["bundle"] = bundle
        _artifacts["explainer"] = explainer
        return bundle, explainer


def main() -> None:
    """Run the serving process."""

    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
