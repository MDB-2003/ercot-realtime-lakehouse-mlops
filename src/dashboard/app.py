"""Houston hub desk: latest silver telemetry, spike probability, and SHAP plots.

Telemetry is the newest row in ``ERCOT_LAKEHOUSE.SILVER.STG_ERCOT_FEATURES``.
Probabilities come from the FastAPI service. Waterfall and force plots are drawn
from ``models/explainer.pkl`` so the figure is the saved TreeExplainer, not a
second model. The probability does not close a breaker.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import requests
import streamlit as st
from dotenv import load_dotenv

from src.api.schemas import SPIKE_FEATURE_COLUMNS

API_BASE = os.environ.get("ERCOT_API_BASE", "http://127.0.0.1:8000").rstrip("/")
EXPLAINER_PATH = REPO_ROOT / "models" / "explainer.pkl"
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
_LABELS = {
    "lmp_hb_houston_usd_mwh": "Houston LMP ($/MWh)",
    "lmp_hb_west_usd_mwh": "West LMP ($/MWh)",
    "operating_reserve_mw": "Operating reserve (MW)",
    "reserve_depletion_velocity_mw_per_min": "Reserve velocity (MW/min)",
    "temperature_f": "Temperature (°F)",
    "heat_index_f": "Heat index (°F)",
    "hour_of_day": "Hour of day (CT)",
    "day_of_week": "Day of week (Mon=0)",
}
_DEFAULTS = {
    "lmp_hb_houston_usd_mwh": 21.0,
    "lmp_hb_west_usd_mwh": 20.0,
    "operating_reserve_mw": 18000.0,
    "reserve_depletion_velocity_mw_per_min": 0.0,
    "temperature_f": 80.0,
    "heat_index_f": 85.0,
    "hour_of_day": 11,
    "day_of_week": 2,
}


def _identifier(value: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"Rejected Snowflake identifier {value!r}")
    return value


def _feature_table() -> str:
    database = _identifier(os.environ.get("SNOWFLAKE_FEATURE_DATABASE", "ERCOT_LAKEHOUSE"))
    schema = _identifier(os.environ.get("SNOWFLAKE_FEATURE_SCHEMA", "SILVER"))
    table = _identifier(os.environ.get("SNOWFLAKE_FEATURE_TABLE", "STG_ERCOT_FEATURES"))
    return f"{database}.{schema}.{table}"


@st.cache_data(ttl=60, show_spinner=False)
def latest_silver_row() -> tuple[dict[str, object] | None, str | None]:
    """Return the newest silver feature row, or an error the page can display."""

    load_dotenv(REPO_ROOT / ".env", override=False)
    required = ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        return None, "Missing Snowflake environment variables: " + ", ".join(missing)
    try:
        import snowflake.connector

        table = _feature_table()
        connection = snowflake.connector.connect(
            account=os.environ["SNOWFLAKE_ACCOUNT"],
            user=os.environ["SNOWFLAKE_USER"],
            password=os.environ["SNOWFLAKE_PASSWORD"],
            warehouse=_identifier(os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")),
            database=_identifier(os.environ.get("SNOWFLAKE_FEATURE_DATABASE", "ERCOT_LAKEHOUSE")),
            schema=_identifier(os.environ.get("SNOWFLAKE_FEATURE_SCHEMA", "SILVER")),
            role=os.environ.get("SNOWFLAKE_ROLE", "").strip() or None,
            client_session_keep_alive=True,
            login_timeout=20,
            network_timeout=30,
            application="ERCOT_Lakehouse",
        )
    except Exception as exc:
        return None, f"Snowflake connection failed ({exc.__class__.__name__})"
    sql = f"""
        SELECT
            sced_timestamp_utc,
            lmp_hb_houston_usd_mwh,
            lmp_hb_west_usd_mwh,
            operating_reserve_mw,
            reserve_depletion_velocity_mw_per_min,
            temperature_f,
            heat_index_f,
            hour_ct AS hour_of_day,
            DAYOFWEEKISO(CONVERT_TIMEZONE('UTC', 'America/Chicago', sced_timestamp_utc)) - 1 AS day_of_week
        FROM {table}
        WHERE lmp_hb_houston_usd_mwh IS NOT NULL
          AND lmp_hb_west_usd_mwh IS NOT NULL
          AND operating_reserve_mw IS NOT NULL
        ORDER BY sced_timestamp_utc DESC
        LIMIT 1
    """
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql)
            row = cursor.fetchone()
            columns = [desc[0].lower() for desc in cursor.description]
    except Exception as exc:
        return None, f"Silver feature read failed ({exc.__class__.__name__})"
    finally:
        connection.close()
    if row is None:
        return None, f"{table} has no priced Houston intervals"
    payload = dict(zip(columns, row, strict=True))
    payload["sced_timestamp_utc"] = str(payload["sced_timestamp_utc"])
    for name in SPIKE_FEATURE_COLUMNS:
        value = payload.get(name)
        payload[name] = None if value is None else float(value)
    return payload, None


def _api_health() -> tuple[bool, str]:
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        response.raise_for_status()
        body = response.json()
    except requests.RequestException as exc:
        return False, f"{API_BASE} is not reachable ({exc.__class__.__name__})"
    loaded = bool(body.get("model_loaded"))
    status = str(body.get("status", "unknown"))
    return loaded, f"{status}; model_loaded={loaded}"


def _controls(defaults: dict[str, float]) -> dict[str, float]:
    for name in SPIKE_FEATURE_COLUMNS:
        st.session_state.setdefault(name, defaults[name])
    values: dict[str, float] = {}
    left, right = st.columns(2)
    for index, name in enumerate(SPIKE_FEATURE_COLUMNS):
        column = left if index % 2 == 0 else right
        with column:
            if name in {"hour_of_day", "day_of_week"}:
                upper = 23 if name == "hour_of_day" else 6
                values[name] = float(st.number_input(_LABELS[name], min_value=0, max_value=upper, step=1, key=name))
            else:
                values[name] = float(st.number_input(_LABELS[name], step=0.01, format="%.4f", key=name))
    return values


def _post_json(path: str, features: dict[str, float], threshold: float) -> tuple[dict[str, object] | None, str | None]:
    params = {"decision_threshold": threshold} if path == "/predict" else None
    try:
        response = requests.post(f"{API_BASE}{path}", json=features, params=params, timeout=30)
    except requests.RequestException as exc:
        return None, f"{path} request failed ({exc.__class__.__name__})"
    if response.status_code != 200:
        detail = response.text[:300]
        return None, f"{path} returned HTTP {response.status_code}: {detail}"
    body = response.json()
    if not isinstance(body, dict):
        return None, f"{path} returned a non-object payload"
    return body, None


def _render_shap(features: dict[str, float]) -> None:
    if not EXPLAINER_PATH.is_file():
        st.error(f"Missing explainer artifact at {EXPLAINER_PATH.name}")
        return
    import matplotlib.pyplot as plt
    import pandas as pd
    import shap

    explainer = _load_explainer()
    frame = pd.DataFrame([features], columns=list(SPIKE_FEATURE_COLUMNS))
    explanation = explainer(frame)
    st.subheader("SHAP waterfall")
    shap.plots.waterfall(explanation[0], max_display=len(SPIKE_FEATURE_COLUMNS), show=False)
    st.pyplot(plt.gcf(), clear_figure=True)
    st.subheader("SHAP force")
    shap.plots.force(explanation[0], matplotlib=True, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)


@st.cache_resource(show_spinner=False)
def _load_explainer() -> object:
    import joblib
    import lightgbm  # noqa: F401
    import shap  # noqa: F401

    return joblib.load(EXPLAINER_PATH)


def main() -> None:
    """Render the desk. Streamlit executes this module on every rerun."""

    st.set_page_config(page_title="HB_HOUSTON Spike Desk", layout="wide")
    st.title("HB_HOUSTON spike desk")
    st.caption("Settlement hub HB_HOUSTON. The spike probability is advisory. Curtailment stays on the dispatch rules.")

    snapshot, source_error = latest_silver_row()
    defaults = dict(_DEFAULTS)
    timestamp = "manual defaults"
    if snapshot is not None:
        timestamp = str(snapshot.get("sced_timestamp_utc", "latest silver row"))
        for name in SPIKE_FEATURE_COLUMNS:
            value = snapshot.get(name)
            if value is not None:
                defaults[name] = float(value)
    if source_error:
        st.warning(source_error)
    elif snapshot is not None and snapshot.get("reserve_depletion_velocity_mw_per_min") is None:
        st.info("Reserve velocity is null on the latest silver row. The score uses the form value.")

    metric_values = snapshot if snapshot is not None else defaults
    st.subheader("Latest telemetry")
    st.caption(timestamp)
    boxes = st.columns(4)
    metric_labels = (
        ("lmp_hb_houston_usd_mwh", "Houston LMP", "$/MWh"),
        ("lmp_hb_west_usd_mwh", "West LMP", "$/MWh"),
        ("operating_reserve_mw", "Operating reserve", "MW"),
        ("reserve_depletion_velocity_mw_per_min", "Reserve velocity", "MW/min"),
    )
    for box, (name, label, unit) in zip(boxes, metric_labels, strict=True):
        raw = None if not isinstance(metric_values, dict) else metric_values.get(name)
        box.metric(label, "—" if raw is None else f"{float(raw):,.2f}", help=unit)

    with st.sidebar:
        st.header("Score")
        reachable, health_text = _api_health()
        st.write(health_text)
        threshold = st.slider("Alert threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.05)
        score_clicked = st.button("Score interval", type="primary", disabled=not reachable)

    st.subheader("Feature vector")
    features = _controls(defaults)
    if score_clicked:
        prediction, predict_error = _post_json("/predict", features, float(threshold))
        explanation, explain_error = _post_json("/explain", features, float(threshold))
        st.session_state["last_score"] = {
            "prediction": prediction,
            "predict_error": predict_error,
            "explanation": explanation,
            "explain_error": explain_error,
            "features": features,
        }

    last = st.session_state.get("last_score")
    if not isinstance(last, dict):
        return
    if last.get("predict_error"):
        st.error(str(last["predict_error"]))
    prediction = last.get("prediction")
    if isinstance(prediction, dict):
        probability = float(prediction["probability_spike_250"])
        alert = bool(prediction["spike_alert"])
        left, right = st.columns(2)
        left.metric("P(LMP > $250/MWh)", f"{probability:.4f}")
        right.metric("Spike alert", "ALERT" if alert else "clear")
    if last.get("explain_error"):
        st.error(str(last["explain_error"]))
    explanation = last.get("explanation")
    if isinstance(explanation, dict) and isinstance(explanation.get("contributions"), list):
        st.subheader("Feature contributions")
        st.dataframe(explanation["contributions"], hide_index=True)
    scored_features = last.get("features")
    if isinstance(scored_features, dict):
        try:
            _render_shap({name: float(scored_features[name]) for name in SPIKE_FEATURE_COLUMNS})
        except Exception as exc:
            st.error(f"Local explainer plot failed ({exc.__class__.__name__})")


main()
