"""Train the HB_HOUSTON spike classifier from the Snowflake silver feature table.

The label is ``IS_SPIKE_250_WITHIN_60MIN``: a Houston LMP above $250 on a later
SCED interval inside the next hour. The same-row price flag is not the target.
Walk-forward ``TimeSeriesSplit`` keeps every test row strictly after its training rows.
``scale_pos_weight`` is the negative-to-positive ratio of that training split
only, so the class weight does not see future scarcity events.

SHAP attributions are on the probability scale. The explainer uses a sample of
the training rows as the interventional background. A model is written only
after PR-AUC, ROC-AUC, and Brier score are defined on at least three test folds
that contain both classes.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from src.api.schemas import SPIKE_FEATURE_COLUMNS, SPIKE_FORECAST_HORIZON_MINUTES, SPIKE_TARGET_COLUMN

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

logger = logging.getLogger("ercot.models.train")

MODEL_NAME = "spike_lgb_v1"
MODEL_VERSION = "1"
N_SPLITS = 5
MIN_BOTH_CLASS_FOLDS = 3
BACKGROUND_ROWS = 200
TARGET_COLUMN = SPIKE_TARGET_COLUMN
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "models" / "spike_lgb_v1.pkl"
EXPLAINER_PATH = REPO_ROOT / "models" / "explainer.pkl"

FEATURE_SQL = """
SELECT
    sced_timestamp_utc,
    lmp_hb_houston_usd_mwh,
    lmp_hb_west_usd_mwh,
    operating_reserve_mw,
    reserve_depletion_velocity_mw_per_min,
    temperature_f,
    heat_index_f,
    hour_ct AS hour_of_day,
    DAYOFWEEKISO(CONVERT_TIMEZONE('UTC', 'America/Chicago', sced_timestamp_utc)) - 1 AS day_of_week,
    is_spike_250_within_60min
FROM {table}
WHERE source = 'ercot_live'
  AND reserve_depletion_velocity_mw_per_min IS NOT NULL
  AND temperature_f IS NOT NULL
  AND heat_index_f IS NOT NULL
  AND hour_ct IS NOT NULL
  AND lmp_hb_houston_usd_mwh IS NOT NULL
  AND lmp_hb_west_usd_mwh IS NOT NULL
  AND operating_reserve_mw IS NOT NULL
  AND is_spike_250_within_60min IS NOT NULL
ORDER BY sced_timestamp_utc
"""


class TrainingDataError(RuntimeError):
    """The silver table cannot support an honest walk-forward spike model."""


def train(
    *,
    model_path: Path = MODEL_PATH,
    explainer_path: Path = EXPLAINER_PATH,
    n_splits: int = N_SPLITS,
) -> dict[str, object]:
    """Fit the classifier, score it walk-forward, and save the model and explainer."""

    frame = load_feature_frame()
    features = frame.loc[:, list(SPIKE_FEATURE_COLUMNS)].astype(float)
    target = frame[TARGET_COLUMN].astype(int).to_numpy()
    _require_trainable_sample(frame, target, n_splits)

    fold_metrics = _cross_validate(features, target, frame["sced_timestamp_utc"], n_splits)
    usable = [row for row in fold_metrics if row["both_classes"]]
    if len(usable) < MIN_BOTH_CLASS_FOLDS:
        raise TrainingDataError(
            f"Only {len(usable)} of {n_splits} test folds contain both classes. "
            "PR-AUC and ROC-AUC are undefined on a single-class window, so the artifact was not saved."
        )
    summary = {
        "pr_auc_mean": float(np.mean([row["pr_auc"] for row in usable])),
        "roc_auc_mean": float(np.mean([row["roc_auc"] for row in usable])),
        "brier_mean": float(np.mean([row["brier"] for row in usable])),
        "folds_scored": len(usable),
        "folds": fold_metrics,
    }
    positives = int(target.sum())
    scale_pos_weight = float((len(target) - positives) / positives)
    model = _new_classifier(scale_pos_weight)
    model.fit(features, target)
    explainer = _fit_explainer(model, features)
    _check_explainer_calibration(model, explainer, features)

    trained_at = datetime.now(timezone.utc).replace(microsecond=0)
    artifact = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "trained_at_utc": trained_at.isoformat(),
        "feature_columns": list(SPIKE_FEATURE_COLUMNS),
        "target": TARGET_COLUMN,
        "horizon_minutes": SPIKE_FORECAST_HORIZON_MINUTES,
        "scale_pos_weight": scale_pos_weight,
        "n_rows": int(len(frame)),
        "n_positives": positives,
        "metrics": summary,
        "model": model,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    joblib.dump(explainer, explainer_path)
    logger.info(
        "Saved %s and %s. PR-AUC %.4f ROC-AUC %.4f Brier %.4f on %s rows (%s positives)",
        model_path,
        explainer_path,
        summary["pr_auc_mean"],
        summary["roc_auc_mean"],
        summary["brier_mean"],
        len(frame),
        positives,
    )
    return {
        "model_path": str(model_path),
        "explainer_path": str(explainer_path),
        "n_rows": int(len(frame)),
        "n_positives": positives,
        "scale_pos_weight": scale_pos_weight,
        "pr_auc_mean": summary["pr_auc_mean"],
        "roc_auc_mean": summary["roc_auc_mean"],
        "brier_mean": summary["brier_mean"],
        "trained_at_utc": trained_at.isoformat(),
    }


def load_feature_frame() -> pd.DataFrame:
    """Read authoritative Houston feature rows from ``ERCOT_LAKEHOUSE.SILVER.STG_ERCOT_FEATURES``."""

    load_environment()
    table = _feature_table_fqn()
    logger.info("Reading spike training rows from %s", table)
    connection = _connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(FEATURE_SQL.format(table=table))
            rows = cursor.fetchall()
            columns = [column[0].lower() for column in cursor.description]
    except Exception as exc:
        raise TrainingDataError(f"Snowflake feature read failed ({exc.__class__.__name__})") from exc
    finally:
        connection.close()

    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        raise TrainingDataError(_empty_feature_reason(table))
    missing = [name for name in (*SPIKE_FEATURE_COLUMNS, TARGET_COLUMN, "sced_timestamp_utc") if name not in frame.columns]
    if missing:
        raise TrainingDataError(f"{table} is missing columns: {', '.join(missing)}")
    frame["sced_timestamp_utc"] = pd.to_datetime(frame["sced_timestamp_utc"])
    frame = frame.sort_values("sced_timestamp_utc").drop_duplicates("sced_timestamp_utc").reset_index(drop=True)
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(int)
    return frame


def _empty_feature_reason(table: str) -> str:
    """Explain an empty training extract using the unfiltered silver counts."""

    diagnosis = "the table has no ercot_live rows with reserve velocity, temperature, and heat index"
    connection = _connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    COUNT(*),
                    COUNT(reserve_depletion_velocity_mw_per_min),
                    COUNT(is_spike_250_within_60min),
                    COALESCE(SUM(IFF(is_spike_250_within_60min = 1, 1, 0)), 0),
                    MIN(lmp_hb_houston_usd_mwh),
                    MAX(lmp_hb_houston_usd_mwh)
                FROM {table}
                WHERE source = 'ercot_live'
                """
            )
            count, with_velocity, labeled, positives, price_min, price_max = cursor.fetchone()
    except Exception:
        count = with_velocity = labeled = positives = price_min = price_max = None
    finally:
        connection.close()
    if count is not None:
        diagnosis = (
            f"{table} has {count} ercot_live rows, {with_velocity} with reserve velocity, "
            f"{labeled} with a complete 60-minute forward window, "
            f"and {positives} IS_SPIKE_250_WITHIN_60MIN positives. "
            f"Houston LMP ranged from {price_min} to {price_max} $/MWh"
        )
    return (
        f"{diagnosis}. The label stays null until 60 minutes of later SCED intervals exist, "
        "and PR-AUC needs both a future spike and a future non-spike. "
        "Same-row prices above $250 are not used as the target. No artifact was saved."
    )


def load_environment() -> None:
    """Load the repository ``.env`` without overriding the process environment."""

    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def _feature_table_fqn() -> str:
    database = _identifier(os.environ.get("SNOWFLAKE_FEATURE_DATABASE", os.environ.get("SNOWFLAKE_DATABASE", "ERCOT_LAKEHOUSE")))
    schema = _identifier(os.environ.get("SNOWFLAKE_FEATURE_SCHEMA", "SILVER"))
    table = _identifier(os.environ.get("SNOWFLAKE_FEATURE_TABLE", "STG_ERCOT_FEATURES"))
    return f"{database}.{schema}.{table}"


def _connect():
    import snowflake.connector

    required = ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise TrainingDataError("Missing Snowflake environment variables: " + ", ".join(missing))
    role = os.environ.get("SNOWFLAKE_ROLE", "").strip()
    warehouse = _identifier(os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"))
    try:
        return snowflake.connector.connect(
            account=os.environ["SNOWFLAKE_ACCOUNT"],
            user=os.environ["SNOWFLAKE_USER"],
            password=os.environ["SNOWFLAKE_PASSWORD"],
            warehouse=warehouse,
            database=_identifier(os.environ.get("SNOWFLAKE_FEATURE_DATABASE", os.environ.get("SNOWFLAKE_DATABASE", "ERCOT_LAKEHOUSE"))),
            schema=_identifier(os.environ.get("SNOWFLAKE_FEATURE_SCHEMA", "SILVER")),
            role=role or None,
            client_session_keep_alive=True,
            login_timeout=60,
            network_timeout=120,
            application="ERCOT_Lakehouse",
        )
    except snowflake.connector.errors.Error as exc:
        raise TrainingDataError(f"Snowflake connection failed ({exc.errno})") from exc


def _require_trainable_sample(frame: pd.DataFrame, target: np.ndarray, n_splits: int) -> None:
    positives = int(target.sum())
    negatives = int(len(target) - positives)
    start = frame["sced_timestamp_utc"].iloc[0]
    end = frame["sced_timestamp_utc"].iloc[-1]
    price_min = float(frame["lmp_hb_houston_usd_mwh"].min())
    price_max = float(frame["lmp_hb_houston_usd_mwh"].max())
    if positives == 0 or negatives == 0:
        raise TrainingDataError(
            f"IS_SPIKE_250_WITHIN_60MIN has {positives} positives and {negatives} negatives "
            f"from {start} to {end}. Houston LMP ranged from {price_min:.2f} to {price_max:.2f} $/MWh. "
            "PR-AUC, ROC-AUC, and scale_pos_weight need both classes, so no artifact was saved."
        )
    if len(target) < n_splits + 1:
        raise TrainingDataError(
            f"TimeSeriesSplit(n_splits={n_splits}) needs at least {n_splits + 1} rows and the feature table has {len(target)}."
        )


def _cross_validate(
    features: pd.DataFrame,
    target: np.ndarray,
    timestamps: pd.Series,
    n_splits: int,
) -> list[dict[str, object]]:
    """Score walk-forward folds with a 60-minute embargo.

    A training label looks up to 60 minutes ahead, so rows whose forward window
    reaches into the test interval are dropped before the fit.
    """

    splitter = TimeSeriesSplit(n_splits=n_splits)
    scored: list[dict[str, object]] = []
    horizon = pd.Timedelta(minutes=SPIKE_FORECAST_HORIZON_MINUTES)
    for fold_index, (train_index, test_index) in enumerate(splitter.split(features)):
        test_start = pd.Timestamp(timestamps.iloc[test_index].min())
        train_ok = pd.to_datetime(timestamps.iloc[train_index]) + horizon < test_start
        train_index = train_index[train_ok.to_numpy()]
        y_train = target[train_index]
        y_test = target[test_index]
        train_positives = int(y_train.sum())
        test_positives = int(y_test.sum())
        row: dict[str, object] = {
            "fold": fold_index,
            "n_train": int(len(train_index)),
            "n_test": int(len(test_index)),
            "n_positives_train": train_positives,
            "n_positives_test": test_positives,
            "both_classes": False,
            "pr_auc": None,
            "roc_auc": None,
            "brier": None,
        }
        if train_positives == 0 or train_positives == len(y_train):
            logger.warning("Fold %s training split has a single class; ranking metrics were skipped", fold_index)
            scored.append(row)
            continue
        if test_positives == 0 or test_positives == len(y_test):
            logger.warning("Fold %s test window has a single class; PR-AUC and ROC-AUC are undefined", fold_index)
            scored.append(row)
            continue
        weight = float((len(y_train) - train_positives) / train_positives)
        model = _new_classifier(weight)
        model.fit(features.iloc[train_index], y_train)
        probability = model.predict_proba(features.iloc[test_index])[:, 1]
        row["both_classes"] = True
        row["pr_auc"] = float(average_precision_score(y_test, probability))
        row["roc_auc"] = float(roc_auc_score(y_test, probability))
        row["brier"] = float(brier_score_loss(y_test, probability))
        scored.append(row)
    return scored


def _new_classifier(scale_pos_weight: float):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=1,
        verbosity=-1,
    )


def _fit_explainer(model, features: pd.DataFrame):
    import shap

    background = features.sample(n=min(BACKGROUND_ROWS, len(features)), random_state=42)
    return shap.TreeExplainer(
        model,
        data=background,
        model_output="probability",
        feature_perturbation="interventional",
    )


def _check_explainer_calibration(model, explainer, features: pd.DataFrame) -> None:
    sample = features.iloc[[0]]
    explanation = explainer(sample)
    values = np.asarray(explanation.values).reshape(-1)
    base = float(np.asarray(explanation.base_values).reshape(-1)[0])
    reconstructed = base + float(values.sum())
    probability = float(model.predict_proba(sample)[0, 1])
    if abs(reconstructed - probability) > 0.05:
        raise TrainingDataError(
            f"SHAP probability reconstruction {reconstructed:.4f} disagrees with predict_proba {probability:.4f}"
        )


def _identifier(value: str) -> str:
    if not _IDENTIFIER.match(value):
        raise TrainingDataError(f"Unsafe Snowflake identifier: {value!r}")
    return value.upper()


def main(argv: list[str] | None = None) -> int:
    """Train from Snowflake and write ``models/spike_lgb_v1.pkl`` plus ``models/explainer.pkl``."""

    parser = argparse.ArgumentParser(description="Train the ERCOT $250/MWh spike classifier.")
    parser.add_argument("--splits", type=int, default=N_SPLITS)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        result = train(n_splits=args.splits)
    except TrainingDataError as exc:
        logger.error("%s", exc)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
