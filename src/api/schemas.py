"""Pydantic v2 contracts for the ERCOT lakehouse, forecaster, and dispatch layer.

Bronze telemetry follows EMIL NP6-788-CD (report type 12300): one SCED row is
``SCEDTimestamp, RepeatedHourFlag, SettlementPoint, LMP``. System reserves, load,
and the Harris County heat index are joined onto that timestamp before landing.

Language-model output is not part of these contracts. Curtailment triggers are
boolean results of deterministic rules and the trained classifier.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

HUB_HOUSTON: Literal["HB_HOUSTON"] = "HB_HOUSTON"
HUB_WEST: Literal["HB_WEST"] = "HB_WEST"
SETTLEMENT_POINTS = (HUB_HOUSTON, HUB_WEST)
SPIKE_THRESHOLDS_USD_MWH = (250.0, 500.0, 1000.0, 5000.0)
SCED_INTERVAL = timedelta(minutes=5)
# Monday is 0, matching datetime.weekday() and Snowflake DAYOFWEEKISO - 1 in America/Chicago.
SPIKE_FEATURE_COLUMNS: tuple[str, ...] = (
    "lmp_hb_houston_usd_mwh",
    "lmp_hb_west_usd_mwh",
    "operating_reserve_mw",
    "reserve_depletion_velocity_mw_per_min",
    "temperature_f",
    "heat_index_f",
    "hour_of_day",
    "day_of_week",
)
# Label is a Houston LMP above $250 at some SCED interval in (t+5min, t+60min].
# The same-row is_spike_250 flag is not a model target: it is a function of the Houston price feature.
SPIKE_TARGET_COLUMN = "is_spike_250_within_60min"
SPIKE_FORECAST_HORIZON_MINUTES = 60
_MW_TOLERANCE = 0.05
_MONEY_TOLERANCE = Decimal("0.01")


class StrictModel(BaseModel):
    """Reject undeclared fields so a drifted payload cannot land silently."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ScedLmpRow(StrictModel):
    """One unmodified settlement-point row from an NP6-788-CD CSV."""

    sced_timestamp_ct: datetime = Field(description="SCEDTimestamp localized to America/Chicago.")
    repeated_hour_flag: Literal["Y", "N"] = Field(description="RepeatedHourFlag. Y is the second DST fallback hour.")
    settlement_point: Literal["HB_HOUSTON", "HB_WEST"]
    lmp_usd_mwh: float = Field(ge=-500.0, le=20_000.0, description="LMP column, dollars per MWh.")

    @field_validator("sced_timestamp_ct")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("sced_timestamp_ct must be timezone-aware")
        return value


class RawErcotTelemetry(StrictModel):
    """Bronze grain: one settlement point at one SCED interval, with system context.

    ``net_load_mw`` is system demand minus wind minus solar. ``authoritative`` is
    true only for rows parsed from ERCOT and NOAA. Simulator rows keep the same
    columns and are marked non-authoritative so a dispatch service can refuse them.
    """

    sced_timestamp_utc: datetime
    interval_start_utc: datetime
    interval_end_utc: datetime
    repeated_hour_flag: Literal["Y", "N"]
    settlement_point: Literal["HB_HOUSTON", "HB_WEST"]
    settlement_point_type: Literal["HU"] = "HU"
    lmp_usd_mwh: float = Field(ge=-500.0, le=20_000.0)
    operating_reserve_mw: float = Field(ge=0.0, le=100_000.0, description="Physical responsive capability, MW.")
    system_demand_mw: float = Field(ge=0.0, le=150_000.0)
    online_capacity_mw: float = Field(ge=0.0, le=200_000.0)
    wind_generation_mw: float = Field(ge=0.0, le=150_000.0)
    solar_generation_mw: float = Field(ge=0.0, le=150_000.0)
    net_load_mw: float = Field(ge=-80_000.0, le=150_000.0)
    eea_level: int | None = Field(default=None, ge=0, le=3)
    grid_state: str | None = Field(default=None, max_length=64)
    heat_index_f: float = Field(ge=-50.0, le=180.0)
    temperature_f: float = Field(ge=-40.0, le=140.0)
    relative_humidity_pct: float = Field(ge=0.0, le=100.0)
    weather_station_id: str = Field(min_length=3, max_length=16)
    weather_observed_at_utc: datetime
    emil_id: str | None = Field(default=None, max_length=32)
    report_type_id: str | None = Field(default=None, max_length=16)
    document_id: str | None = Field(default=None, max_length=32)
    published_at_utc: datetime | None = None
    source: Literal["ercot_live", "simulator"]
    authoritative: bool
    fallback_reason: str | None = Field(default=None, max_length=4000)
    ingested_at_utc: datetime

    @field_validator(
        "sced_timestamp_utc",
        "interval_start_utc",
        "interval_end_utc",
        "weather_observed_at_utc",
        "published_at_utc",
        "ingested_at_utc",
    )
    @classmethod
    def _utc_aware(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("telemetry timestamps must be timezone-aware UTC")
        return value

    @model_validator(mode="after")
    def _check_identity(self) -> RawErcotTelemetry:
        expected_net_load = self.system_demand_mw - self.wind_generation_mw - self.solar_generation_mw
        if abs(expected_net_load - self.net_load_mw) > _MW_TOLERANCE:
            raise ValueError("net_load_mw must equal system_demand_mw - wind_generation_mw - solar_generation_mw")
        if self.interval_end_utc - self.interval_start_utc != SCED_INTERVAL:
            raise ValueError("interval_end_utc must be five minutes after interval_start_utc")
        if not (self.interval_start_utc <= self.sced_timestamp_utc < self.interval_end_utc):
            raise ValueError("sced_timestamp_utc must fall inside the five-minute interval")
        if self.source == "simulator":
            if self.authoritative:
                raise ValueError("simulator telemetry cannot be authoritative")
            if not self.fallback_reason:
                raise ValueError("simulator telemetry requires fallback_reason")
            if self.emil_id is not None or self.document_id is not None:
                raise ValueError("simulator telemetry must not carry an EMIL document identity")
        if self.source == "ercot_live":
            if not self.authoritative:
                raise ValueError("live telemetry must be authoritative")
            if self.fallback_reason is not None:
                raise ValueError("live telemetry must not carry fallback_reason")
            if self.emil_id != "NP6-788-CD" or self.report_type_id != "12300" or not self.document_id:
                raise ValueError("live telemetry requires NP6-788-CD report type 12300 and a document id")
        if self.grid_state is not None and not self.grid_state.strip():
            raise ValueError("grid_state cannot be blank")
        return self


class FeatureSnapshot(StrictModel):
    """Silver features presented to the spike classifier at decision time."""

    as_of_utc: datetime
    settlement_point: Literal["HB_HOUSTON"] = HUB_HOUSTON
    lmp_usd_mwh: float = Field(ge=-500.0, le=20_000.0)
    lmp_hb_west_usd_mwh: float = Field(ge=-500.0, le=20_000.0)
    nodal_congestion_spread_usd_mwh: float
    operating_reserve_mw: float = Field(ge=0.0, le=100_000.0)
    reserve_depletion_velocity_mw_per_min: float
    system_demand_mw: float = Field(ge=0.0, le=150_000.0)
    net_load_mw: float
    net_load_acceleration_mw_per_min2: float
    heat_index_f: float = Field(ge=-50.0, le=180.0)
    hour_ct: int = Field(ge=0, le=23)
    eea_level: int = Field(ge=0, le=3)

    @field_validator("as_of_utc")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("as_of_utc must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _spread_matches_hubs(self) -> FeatureSnapshot:
        expected = self.lmp_usd_mwh - self.lmp_hb_west_usd_mwh
        if abs(expected - self.nodal_congestion_spread_usd_mwh) > 0.01:
            raise ValueError("nodal_congestion_spread_usd_mwh must equal Houston LMP minus West LMP")
        return self


class InferenceRequest(StrictModel):
    """Request for P(LMP > $250/MWh) over a 60 to 120 minute horizon.

    ``decision_threshold`` is applied by the serving layer after the model
    returns a probability. It is not an instruction to an LLM.
    """

    request_id: UUID = Field(default_factory=uuid4)
    requested_at_utc: datetime
    horizon_minutes: int = Field(ge=60, le=120)
    decision_threshold: float = Field(default=0.50, gt=0.0, lt=1.0)
    features: FeatureSnapshot

    @field_validator("requested_at_utc")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("requested_at_utc must be timezone-aware")
        return value


class FeatureAttribution(StrictModel):
    """One TreeExplainer contribution for a single prediction."""

    rank: int = Field(ge=1, le=3)
    feature_name: str = Field(min_length=1, max_length=128)
    feature_value: float
    shap_value: float
    abs_shap_value: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _absolute_value_matches(self) -> FeatureAttribution:
        if abs(self.abs_shap_value - abs(self.shap_value)) > 1e-9:
            raise ValueError("abs_shap_value must equal abs(shap_value)")
        return self


class ShapAttributionPayload(StrictModel):
    """Top three SHAP attributions for a 60-minute-ahead spike probability."""

    prediction_id: UUID = Field(default_factory=uuid4)
    model_name: str = Field(min_length=1, max_length=128)
    model_version: str = Field(min_length=1, max_length=64)
    as_of_utc: datetime
    horizon_minutes: Literal[60] = SPIKE_FORECAST_HORIZON_MINUTES
    probability_spike_250: float = Field(ge=0.0, le=1.0)
    predicted_positive: bool
    decision_threshold: float = Field(gt=0.0, lt=1.0)
    base_value: float
    top_features: list[FeatureAttribution] = Field(min_length=3, max_length=3)

    @field_validator("as_of_utc")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("as_of_utc must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _ranking_and_decision(self) -> ShapAttributionPayload:
        ranks = [item.rank for item in self.top_features]
        if ranks != [1, 2, 3]:
            raise ValueError("top_features must be ranked 1, 2, 3 in that order")
        magnitudes = [item.abs_shap_value for item in self.top_features]
        if magnitudes != sorted(magnitudes, reverse=True):
            raise ValueError("top_features must be ordered by descending absolute SHAP value")
        expected_positive = self.probability_spike_250 >= self.decision_threshold
        if self.predicted_positive is not expected_positive:
            raise ValueError("predicted_positive must equal probability_spike_250 >= decision_threshold")
        return self


class SpikeFeatureVector(StrictModel):
    """Columns the LightGBM spike classifier was trained on.

    ``day_of_week`` is America/Chicago with Monday = 0. SHAP values returned
    with a forecast are on the probability scale, not the log-odds margin.
    """

    lmp_hb_houston_usd_mwh: float = Field(ge=-500.0, le=20_000.0)
    lmp_hb_west_usd_mwh: float = Field(ge=-500.0, le=20_000.0)
    operating_reserve_mw: float = Field(ge=0.0, le=100_000.0)
    reserve_depletion_velocity_mw_per_min: float
    temperature_f: float = Field(ge=-40.0, le=140.0)
    heat_index_f: float = Field(ge=-50.0, le=180.0)
    hour_of_day: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)


class PredictResponse(StrictModel):
    """Probability that HB_HOUSTON exceeds $250/MWh within the next 60 minutes."""

    model_name: str = Field(min_length=1, max_length=128)
    probability_spike_250: float = Field(ge=0.0, le=1.0)
    spike_alert: bool
    decision_threshold: float = Field(gt=0.0, lt=1.0)
    horizon_minutes: Literal[60] = SPIKE_FORECAST_HORIZON_MINUTES

    @model_validator(mode="after")
    def _alert_matches_threshold(self) -> PredictResponse:
        expected = self.probability_spike_250 >= self.decision_threshold
        if self.spike_alert is not expected:
            raise ValueError("spike_alert must equal probability_spike_250 >= decision_threshold")
        return self


class FeatureContribution(StrictModel):
    """One probability-scale SHAP contribution, ranked across the full feature vector."""

    rank: int = Field(ge=1, le=32)
    feature_name: str = Field(min_length=1, max_length=128)
    feature_value: float
    shap_value: float
    abs_shap_value: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _absolute_value_matches(self) -> FeatureContribution:
        if abs(self.abs_shap_value - abs(self.shap_value)) > 1e-6:
            raise ValueError("abs_shap_value must equal abs(shap_value)")
        return self


class ExplainResponse(StrictModel):
    """Every TreeExplainer contribution for one feature vector, largest magnitude first."""

    model_name: str = Field(min_length=1, max_length=128)
    probability_spike_250: float = Field(ge=0.0, le=1.0)
    horizon_minutes: Literal[60] = SPIKE_FORECAST_HORIZON_MINUTES
    base_value: float
    contributions: list[FeatureContribution] = Field(min_length=1)

    @model_validator(mode="after")
    def _complete_and_sorted(self) -> ExplainResponse:
        names = [item.feature_name for item in self.contributions]
        if set(names) != set(SPIKE_FEATURE_COLUMNS) or len(names) != len(SPIKE_FEATURE_COLUMNS):
            raise ValueError("contributions must cover each spike feature once")
        ranks = [item.rank for item in self.contributions]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("contributions must be ranked 1..n in order")
        magnitudes = [item.abs_shap_value for item in self.contributions]
        if magnitudes != sorted(magnitudes, reverse=True):
            raise ValueError("contributions must be ordered by descending absolute SHAP value")
        return self


class ForecastRequest(StrictModel):
    """Current telemetry vector for P(HB_HOUSTON > $250/MWh within 60 minutes)."""

    as_of_utc: datetime
    horizon_minutes: Literal[60] = SPIKE_FORECAST_HORIZON_MINUTES
    decision_threshold: float = Field(default=0.50, gt=0.0, lt=1.0)
    features: SpikeFeatureVector

    @field_validator("as_of_utc")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("as_of_utc must be timezone-aware")
        return value


class DispatchStrategy(str, Enum):
    NO_INTERVENTION = "no_intervention"
    AI_TRIGGERED_CURTAILMENT = "ai_triggered_curtailment"
    BESS_PLUS_CURTAILMENT = "bess_plus_curtailment"


class IntervalDispatch(StrictModel):
    """Deterministic position for one five-minute settlement interval."""

    interval_start_utc: datetime
    interval_end_utc: datetime
    lmp_usd_mwh: float
    spike_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    curtailment_triggered: bool
    load_served_mw: float = Field(ge=0.0, le=50.0)
    curtailed_mw: float = Field(ge=0.0, le=20.0)
    bess_discharge_mw: float = Field(ge=0.0, le=10.0)
    bess_soc_mwh_end: float = Field(ge=0.0, le=20.0)
    energy_cost_usd: Decimal
    downtime_cost_usd: Decimal = Field(ge=0)

    @model_validator(mode="after")
    def _interval_and_balance(self) -> IntervalDispatch:
        if self.interval_start_utc.tzinfo is None or self.interval_end_utc.tzinfo is None:
            raise ValueError("dispatch interval timestamps must be timezone-aware")
        if self.interval_end_utc - self.interval_start_utc != SCED_INTERVAL:
            raise ValueError("dispatch intervals are five minutes")
        # Grid import plus battery discharge plus shed load equals the 50 MW baseline.
        expected_grid_mw = 50.0 - self.curtailed_mw - self.bess_discharge_mw
        if expected_grid_mw < -_MW_TOLERANCE:
            raise ValueError("curtailment plus battery discharge cannot exceed the 50 MW baseline")
        if abs(self.load_served_mw - expected_grid_mw) > _MW_TOLERANCE:
            raise ValueError("load_served_mw must equal 50 MW minus curtailment minus battery discharge")
        if self.curtailment_triggered and self.curtailed_mw <= 0:
            raise ValueError("a triggered interval must shed discretionary load")
        if not self.curtailment_triggered and self.curtailed_mw != 0:
            raise ValueError("an interval that is not triggered cannot shed load")
        return self


class StrategyOutcome(StrictModel):
    """Plant-level cost of one dispatch policy over the simulated horizon."""

    strategy: DispatchStrategy
    intervals: list[IntervalDispatch] = Field(min_length=1)
    energy_spend_usd: Decimal
    downtime_cost_usd: Decimal = Field(ge=0)
    total_cost_usd: Decimal
    avoided_energy_spend_usd: Decimal
    curtailed_mwh: Decimal = Field(ge=0)
    bess_discharged_mwh: Decimal = Field(ge=0)
    peak_facility_load_mw: float = Field(ge=0.0, le=50.0)
    four_cp_exposure_mw: float = Field(ge=0.0, le=50.0)
    four_cp_tariff_risk_usd: Decimal = Field(ge=0)

    @model_validator(mode="after")
    def _totals_match_intervals(self) -> StrategyOutcome:
        energy = sum((row.energy_cost_usd for row in self.intervals), Decimal("0"))
        downtime = sum((row.downtime_cost_usd for row in self.intervals), Decimal("0"))
        if abs(energy - self.energy_spend_usd) > _MONEY_TOLERANCE:
            raise ValueError("energy_spend_usd must equal the sum of interval energy costs")
        if abs(downtime - self.downtime_cost_usd) > _MONEY_TOLERANCE:
            raise ValueError("downtime_cost_usd must equal the sum of interval downtime costs")
        if abs((self.energy_spend_usd + self.downtime_cost_usd) - self.total_cost_usd) > _MONEY_TOLERANCE:
            raise ValueError("total_cost_usd must equal energy spend plus downtime cost")
        hours = Decimal(len(self.intervals)) * Decimal("5") / Decimal("60")
        expected_curtailed = sum((Decimal(str(row.curtailed_mw)) for row in self.intervals), Decimal("0")) * (
            Decimal("5") / Decimal("60")
        )
        if hours <= 0:
            raise ValueError("strategy requires a positive horizon")
        if abs(expected_curtailed - self.curtailed_mwh) > Decimal("0.05"):
            raise ValueError("curtailed_mwh must equal the time-integral of curtailed_mw")
        return self


class DispatchSimulationResult(StrictModel):
    """Comparison of no intervention, curtailment, and curtailment plus the LFP battery.

    ``llm_authorized_trigger`` is fixed false. A narrative briefing cannot set
    ``curtailment_triggered`` on any interval.
    """

    simulation_id: UUID = Field(default_factory=uuid4)
    facility_name: str = Field(min_length=1, max_length=200)
    settlement_point: Literal["HB_HOUSTON"] = HUB_HOUSTON
    generated_at_utc: datetime
    trigger_rule: Literal["deterministic_classifier_threshold"] = "deterministic_classifier_threshold"
    llm_authorized_trigger: Literal[False] = False
    outcomes: list[StrategyOutcome] = Field(min_length=3, max_length=3)
    lowest_cost_strategy: DispatchStrategy

    @model_validator(mode="after")
    def _strategies_and_avoided_spend(self) -> DispatchSimulationResult:
        strategies = [item.strategy for item in self.outcomes]
        expected = list(DispatchStrategy)
        if sorted(strategies, key=lambda item: item.value) != sorted(expected, key=lambda item: item.value):
            raise ValueError("outcomes must contain each dispatch strategy exactly once")
        baseline = next(item for item in self.outcomes if item.strategy is DispatchStrategy.NO_INTERVENTION)
        for item in self.outcomes:
            avoided = baseline.energy_spend_usd - item.energy_spend_usd
            if item.strategy is DispatchStrategy.NO_INTERVENTION:
                avoided = Decimal("0")
            if abs(item.avoided_energy_spend_usd - avoided) > _MONEY_TOLERANCE:
                raise ValueError("avoided_energy_spend_usd must be measured against no intervention")
            if item.strategy is DispatchStrategy.NO_INTERVENTION and item.curtailed_mwh != 0:
                raise ValueError("no intervention cannot curtail")
            if item.strategy is not DispatchStrategy.BESS_PLUS_CURTAILMENT and item.bess_discharged_mwh != 0:
                raise ValueError("battery discharge is only valid for bess_plus_curtailment")
        costs = {item.strategy: item.total_cost_usd for item in self.outcomes}
        best = min(costs.values())
        if costs[self.lowest_cost_strategy] != best:
            raise ValueError("lowest_cost_strategy must be a minimum total_cost_usd strategy")
        return self


class DispatchIntervalInput(StrictModel):
    """One five-minute price and spike probability for the dispatch engine."""

    interval_start_utc: datetime
    lmp_usd_mwh: float = Field(ge=-500.0, le=20_000.0)
    spike_probability: float = Field(ge=0.0, le=1.0)

    @field_validator("interval_start_utc")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("interval_start_utc must be timezone-aware")
        return value


class DispatchSimulationRequest(StrictModel):
    """Horizon the deterministic engine turns into a curtailment schedule.

    Physical action follows ``spike_probability > decision_threshold``.
    A language-model score is not an input.
    """

    intervals: list[DispatchIntervalInput] = Field(min_length=1, max_length=288)
    decision_threshold: float = Field(default=0.50, gt=0.0, lt=1.0)
    initial_soc_mwh: float | None = Field(default=None, ge=0.0, le=20.0)


class StrategySavings(StrictModel):
    """Dollar comparison of one policy against unmitigated baseline exposure."""

    strategy: DispatchStrategy
    energy_spend_usd: Decimal
    downtime_penalty_usd: Decimal = Field(ge=0)
    total_cost_usd: Decimal
    avoided_energy_spend_usd: Decimal
    net_savings_vs_baseline_usd: Decimal
    four_cp_tariff_risk_usd: Decimal = Field(ge=0)
    four_cp_risk_mitigation_usd: Decimal


class GridSimulationResponse(StrictModel):
    """Dispatch schedule plus the savings matrix for the three policies."""

    schedule: DispatchSimulationResult
    savings_matrix: list[StrategySavings] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def _matrix_matches_schedule(self) -> GridSimulationResponse:
        by_strategy = {item.strategy: item for item in self.schedule.outcomes}
        if [row.strategy for row in self.savings_matrix] != [item.strategy for item in self.schedule.outcomes]:
            raise ValueError("savings_matrix must follow the schedule outcome order")
        baseline = by_strategy[DispatchStrategy.NO_INTERVENTION]
        for row in self.savings_matrix:
            outcome = by_strategy[row.strategy]
            if row.energy_spend_usd != outcome.energy_spend_usd:
                raise ValueError("savings energy spend must match the schedule")
            if row.downtime_penalty_usd != outcome.downtime_cost_usd:
                raise ValueError("savings downtime penalty must match the schedule")
            if row.total_cost_usd != outcome.total_cost_usd:
                raise ValueError("savings total cost must match the schedule")
            if row.avoided_energy_spend_usd != outcome.avoided_energy_spend_usd:
                raise ValueError("savings avoided energy must match the schedule")
            if row.four_cp_tariff_risk_usd != outcome.four_cp_tariff_risk_usd:
                raise ValueError("savings 4CP risk must match the schedule")
            net = baseline.total_cost_usd - outcome.total_cost_usd
            mitigation = baseline.four_cp_tariff_risk_usd - outcome.four_cp_tariff_risk_usd
            if row.strategy is DispatchStrategy.NO_INTERVENTION:
                net = Decimal("0")
                mitigation = Decimal("0")
            if abs(row.net_savings_vs_baseline_usd - net) > _MONEY_TOLERANCE:
                raise ValueError("net savings must equal baseline total cost minus strategy total cost")
            if abs(row.four_cp_risk_mitigation_usd - mitigation) > _MONEY_TOLERANCE:
                raise ValueError("4CP mitigation must equal baseline tariff risk minus strategy tariff risk")
        return self


class WalkForwardFold(StrictModel):
    """One TimeSeriesSplit / walk-forward evaluation window."""

    fold_index: int = Field(ge=0)
    train_start_utc: datetime
    train_end_utc: datetime
    test_start_utc: datetime
    test_end_utc: datetime
    n_train: int = Field(ge=1)
    n_test: int = Field(ge=1)
    n_positives_test: int = Field(ge=0)
    pr_auc: float = Field(ge=0.0, le=1.0)
    brier_score: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _chronology(self) -> WalkForwardFold:
        if not (self.train_start_utc < self.train_end_utc <= self.test_start_utc < self.test_end_utc):
            raise ValueError("fold windows must be chronological and non-overlapping")
        if self.n_positives_test > self.n_test:
            raise ValueError("n_positives_test cannot exceed n_test")
        return self


class StrategyBacktestPnl(StrictModel):
    """Realized simulation PnL of one policy on the concatenated test folds."""

    strategy: DispatchStrategy
    total_energy_spend_usd: Decimal
    total_downtime_cost_usd: Decimal = Field(ge=0)
    total_cost_usd: Decimal
    avoided_energy_spend_vs_baseline_usd: Decimal
    four_cp_tariff_risk_usd: Decimal = Field(ge=0)
    spike_intervals: int = Field(ge=0)
    intervals_curtailed: int = Field(ge=0)

    @model_validator(mode="after")
    def _cost_identity(self) -> StrategyBacktestPnl:
        expected = self.total_energy_spend_usd + self.total_downtime_cost_usd
        if abs(expected - self.total_cost_usd) > _MONEY_TOLERANCE:
            raise ValueError("total_cost_usd must equal energy spend plus downtime")
        if self.intervals_curtailed > 0 and self.strategy is DispatchStrategy.NO_INTERVENTION:
            raise ValueError("no intervention cannot record curtailed intervals")
        return self


class BacktestReport(StrictModel):
    """Walk-forward classification metrics plus the economic replay of each policy."""

    report_id: UUID = Field(default_factory=uuid4)
    generated_at_utc: datetime
    model_name: str = Field(min_length=1, max_length=128)
    target_flag: Literal["is_spike_250", "is_spike_500", "is_spike_1000", "is_spike_5000"]
    settlement_point: Literal["HB_HOUSTON"] = HUB_HOUSTON
    splitter: Literal["TimeSeriesSplit", "walk_forward"]
    folds: list[WalkForwardFold] = Field(min_length=2)
    mean_pr_auc: float = Field(ge=0.0, le=1.0)
    mean_brier_score: float = Field(ge=0.0, le=1.0)
    strategy_results: list[StrategyBacktestPnl] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def _aggregate_matches_folds(self) -> BacktestReport:
        indexes = [fold.fold_index for fold in self.folds]
        if indexes != list(range(len(self.folds))):
            raise ValueError("folds must be indexed 0..n-1 in walk-forward order")
        for earlier, later in zip(self.folds, self.folds[1:]):
            if later.test_start_utc < earlier.test_end_utc:
                raise ValueError("test folds must not overlap")
        mean_pr = sum(fold.pr_auc for fold in self.folds) / len(self.folds)
        mean_brier = sum(fold.brier_score for fold in self.folds) / len(self.folds)
        if abs(mean_pr - self.mean_pr_auc) > 1e-6:
            raise ValueError("mean_pr_auc must equal the unweighted mean of fold PR-AUC")
        if abs(mean_brier - self.mean_brier_score) > 1e-6:
            raise ValueError("mean_brier_score must equal the unweighted mean of fold Brier scores")
        strategies = [item.strategy for item in self.strategy_results]
        if len(set(strategies)) != 3:
            raise ValueError("strategy_results must contain each dispatch strategy once")
        baseline = next(
            item for item in self.strategy_results if item.strategy is DispatchStrategy.NO_INTERVENTION
        )
        if baseline.avoided_energy_spend_vs_baseline_usd != 0:
            raise ValueError("baseline avoided energy spend must be zero")
        for item in self.strategy_results:
            avoided = baseline.total_energy_spend_usd - item.total_energy_spend_usd
            if item.strategy is DispatchStrategy.NO_INTERVENTION:
                continue
            if abs(item.avoided_energy_spend_vs_baseline_usd - avoided) > _MONEY_TOLERANCE:
                raise ValueError("strategy avoided spend must equal baseline energy spend minus strategy energy spend")
        return self


class WeatherStation(StrictModel):
    latitude: float = Field(ge=29.0, le=30.5)
    longitude: float = Field(ge=-96.0, le=-94.0)
    county: Literal["Harris"]
    nws_grid: Literal["HGX"]


class PowerLimits(StrictModel):
    total_baseline_load_mw: float = Field(ge=0)
    critical_uninterruptible_load_mw: float = Field(ge=0)
    max_curtailable_load_mw: float = Field(ge=0)


class FacilityEconomics(StrictModel):
    hourly_production_downtime_cost_usd: float = Field(gt=0)
    scarcity_threshold_usd_mwh: float = Field(gt=0)
    spike_event_thresholds_usd_mwh: list[float] = Field(min_length=4, max_length=4)
    avoided_4cp_tariff_usd_per_mw_year: float = Field(ge=0)


class BatteryStorage(StrictModel):
    chemistry: Literal["LFP"]
    chemistry_name: Literal["Lithium-iron-phosphate"]
    rated_power_mw: float = Field(gt=0)
    rated_energy_mwh: float = Field(gt=0)
    max_discharge_power_mw: float = Field(gt=0)
    max_charge_power_mw: float = Field(gt=0)
    roundtrip_efficiency: float = Field(gt=0, le=1)
    min_soc_fraction: float = Field(ge=0, lt=1)
    max_soc_fraction: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def _soc_window(self) -> BatteryStorage:
        if self.min_soc_fraction >= self.max_soc_fraction:
            raise ValueError("min_soc_fraction must be below max_soc_fraction")
        return self


class CurtailableAsset(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    shed_capacity_mw: float = Field(gt=0, le=20)
    max_duration_hours: float = Field(gt=0, le=24)


class DispatchPolicy(StrictModel):
    primary_scarcity_threshold_usd_mwh: float = Field(gt=0)
    forecast_horizon_minutes_min: int = Field(ge=60, le=120)
    forecast_horizon_minutes_max: int = Field(ge=60, le=120)
    llm_may_authorize_curtailment: Literal[False]


class FacilityProfile(StrictModel):
    """Typed view of ``config/facility_profile.yaml``."""

    name: str
    industry: str
    site: Literal["Houston Ship Channel"]
    county: Literal["Harris"]
    state: Literal["TX"]
    utility_interconnect: str
    location_hub: Literal["HB_HOUSTON"]
    congestion_reference_hub: Literal["HB_WEST"]
    weather_station: WeatherStation
    power_limits: PowerLimits
    critical_process_loads: list[str] = Field(min_length=3, max_length=3)
    economics: FacilityEconomics
    battery_storage: BatteryStorage
    curtailable_assets: list[CurtailableAsset] = Field(min_length=3, max_length=3)
    dispatch_policy: DispatchPolicy

    @model_validator(mode="after")
    def _engineering_contract(self) -> FacilityProfile:
        limits = self.power_limits
        shed = sum(asset.shed_capacity_mw for asset in self.curtailable_assets)
        if not math.isclose(shed, limits.max_curtailable_load_mw, abs_tol=1e-6):
            raise ValueError("curtailable asset capacities must sum to max_curtailable_load_mw")
        if not math.isclose(
            limits.critical_uninterruptible_load_mw + shed,
            limits.total_baseline_load_mw,
            abs_tol=1e-6,
        ):
            raise ValueError("critical load plus curtailable load must equal total baseline load")
        if not math.isclose(limits.total_baseline_load_mw, 50.0, abs_tol=1e-6):
            raise ValueError("total baseline load is 50.0 MW")
        if not math.isclose(limits.critical_uninterruptible_load_mw, 30.0, abs_tol=1e-6):
            raise ValueError("critical uninterruptible load is 30.0 MW")
        expected_assets = {
            "Air compressors": 10.0,
            "Polymer pelletizers": 6.0,
            "Secondary chillers": 4.0,
        }
        actual_assets = {asset.name: asset.shed_capacity_mw for asset in self.curtailable_assets}
        if actual_assets != expected_assets:
            raise ValueError("curtailable assets must be the 10/6/4 MW compressor, pelletizer, and chiller set")
        if self.critical_process_loads != ["Reactors", "Catalytic cooling loops", "Flares"]:
            raise ValueError("critical process loads must be reactors, catalytic cooling loops, and flares")
        if self.economics.spike_event_thresholds_usd_mwh != list(SPIKE_THRESHOLDS_USD_MWH):
            raise ValueError("spike thresholds must be 250, 500, 1000, and 5000 $/MWh")
        if not math.isclose(self.economics.scarcity_threshold_usd_mwh, 250.0, abs_tol=1e-6):
            raise ValueError("primary scarcity threshold is 250 $/MWh")
        if not math.isclose(self.economics.hourly_production_downtime_cost_usd, 6500.0, abs_tol=1e-6):
            raise ValueError("production downtime cost is 6500 $/hr")
        battery = self.battery_storage
        if not math.isclose(battery.rated_power_mw, 10.0, abs_tol=1e-6):
            raise ValueError("BESS nameplate power is 10.0 MW")
        if not math.isclose(battery.rated_energy_mwh, 20.0, abs_tol=1e-6):
            raise ValueError("BESS nameplate energy is 20.0 MWh")
        if battery.chemistry != "LFP":
            raise ValueError("BESS chemistry is LFP")
        if self.dispatch_policy.forecast_horizon_minutes_min > self.dispatch_policy.forecast_horizon_minutes_max:
            raise ValueError("forecast horizon minimum cannot exceed the maximum")
        if self.dispatch_policy.llm_may_authorize_curtailment:
            raise ValueError("the language model cannot authorize curtailment")
        return self


def default_facility_profile_path() -> Path:
    """Repository ``config/facility_profile.yaml``, independent of the process cwd."""

    return Path(__file__).resolve().parents[2] / "config" / "facility_profile.yaml"


def load_facility_profile(path: Path | None = None) -> FacilityProfile:
    """Load and validate the ship-channel facility contract."""

    profile_path = path or default_facility_profile_path()
    try:
        document = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FileNotFoundError(f"Facility profile not found at {profile_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Facility profile YAML is invalid: {profile_path}") from exc
    if not isinstance(document, dict) or "facility" not in document:
        raise ValueError("Facility profile must contain a top-level 'facility' mapping")
    return FacilityProfile.model_validate(document["facility"])
