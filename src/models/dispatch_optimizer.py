"""Deterministic 5-minute dispatch for the Houston Ship Channel olefins unit.

Three policies are scored on the same price path. None of them reads a
language-model decision.

* Unmitigated baseline buys the full 50 MW at the hub LMP.
* Automated curtailment sheds discretionary load, up to 20 MW, when
  ``P(spike) > threshold``. Asset duration limits still apply.
* Co-optimized dispatch does that shed and also discharges the 10 MW / 20 MWh
  LFP battery, down to the 15% state-of-charge floor.

Downtime is charged in proportion to the megawatts shed. Shedding the full
20 MW costs the profile's $6,500 per hour. Battery discharge does not, because
the process load stays served. Energy cost is grid import times the interval
length times LMP.

4CP tariff risk is ``peak grid import during the horizon × $54,000 per MW-year``.
That is the annual transmission exposure if this horizon sets a summer
coincident peak. It is not a bill for the simulated hours.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP

from src.api.schemas import (
    SCED_INTERVAL,
    DispatchIntervalInput,
    DispatchSimulationRequest,
    DispatchSimulationResult,
    DispatchStrategy,
    FacilityProfile,
    GridSimulationResponse,
    IntervalDispatch,
    StrategyOutcome,
    StrategySavings,
    load_facility_profile,
)

_CENTS = Decimal("0.01")
_MW = Decimal("0.001")
_INTERVAL_HOURS = Decimal(5) / Decimal(60)


def simulate_dispatch(
    request: DispatchSimulationRequest,
    profile: FacilityProfile | None = None,
) -> GridSimulationResponse:
    """Score baseline, curtailment, and battery-plus-curtailment on one horizon."""

    facility = profile or load_facility_profile()
    intervals = _ordered_intervals(request.intervals)
    initial_soc = _initial_soc(request.initial_soc_mwh, facility)
    outcomes = _with_avoided_energy(
        [
            _run_strategy(strategy, intervals, request.decision_threshold, initial_soc, facility)
            for strategy in DispatchStrategy
        ]
    )
    best = min(outcomes, key=lambda item: (item.total_cost_usd, _strategy_rank(item.strategy)))
    schedule = DispatchSimulationResult(
        facility_name=facility.name,
        settlement_point="HB_HOUSTON",
        generated_at_utc=datetime.now(timezone.utc).replace(microsecond=0),
        outcomes=outcomes,
        lowest_cost_strategy=best.strategy,
    )
    return GridSimulationResponse(schedule=schedule, savings_matrix=_savings_matrix(schedule))


def _run_strategy(
    strategy: DispatchStrategy,
    intervals: list[DispatchIntervalInput],
    threshold: float,
    initial_soc: Decimal,
    profile: FacilityProfile,
) -> StrategyOutcome:
    soc = initial_soc
    hours_used = {asset.name: Decimal("0") for asset in profile.curtailable_assets}
    rows: list[IntervalDispatch] = []
    for interval in intervals:
        row, soc, hours_used = _interval_position(strategy, interval, threshold, soc, hours_used, profile)
        rows.append(row)
    energy = sum((row.energy_cost_usd for row in rows), Decimal("0"))
    downtime = sum((row.downtime_cost_usd for row in rows), Decimal("0"))
    peak = max(row.load_served_mw for row in rows)
    rate = Decimal(str(profile.economics.avoided_4cp_tariff_usd_per_mw_year))
    return StrategyOutcome(
        strategy=strategy,
        intervals=rows,
        energy_spend_usd=energy,
        downtime_cost_usd=downtime,
        total_cost_usd=energy + downtime,
        avoided_energy_spend_usd=Decimal("0"),
        curtailed_mwh=_integrated_mw(row.curtailed_mw for row in rows),
        bess_discharged_mwh=_integrated_mw(row.bess_discharge_mw for row in rows),
        peak_facility_load_mw=peak,
        four_cp_exposure_mw=peak,
        four_cp_tariff_risk_usd=_money(Decimal(str(peak)) * rate),
    )


def _interval_position(
    strategy: DispatchStrategy,
    interval: DispatchIntervalInput,
    threshold: float,
    soc: Decimal,
    hours_used: dict[str, Decimal],
    profile: FacilityProfile,
) -> tuple[IntervalDispatch, Decimal, dict[str, Decimal]]:
    act = interval.spike_probability > threshold
    curtailed = Decimal("0")
    next_hours = dict(hours_used)
    discharge = Decimal("0")
    next_soc = soc
    if act and strategy in {DispatchStrategy.AI_TRIGGERED_CURTAILMENT, DispatchStrategy.BESS_PLUS_CURTAILMENT}:
        curtailed, next_hours = _allocate_shed(hours_used, profile)
    if act and strategy is DispatchStrategy.BESS_PLUS_CURTAILMENT:
        discharge, next_soc = _allocate_discharge(soc, profile)
    curtailed_mw = float(curtailed)
    discharge_mw = float(discharge)
    grid_mw = float(profile.power_limits.total_baseline_load_mw) - curtailed_mw - discharge_mw
    start = interval.interval_start_utc.astimezone(timezone.utc)
    energy = _money(Decimal(str(grid_mw)) * _INTERVAL_HOURS * Decimal(str(interval.lmp_usd_mwh)))
    max_shed = Decimal(str(profile.power_limits.max_curtailable_load_mw))
    hourly_downtime = Decimal(str(profile.economics.hourly_production_downtime_cost_usd))
    penalty = _money(hourly_downtime * (curtailed / max_shed) * _INTERVAL_HOURS) if curtailed > 0 else Decimal("0")
    row = IntervalDispatch(
        interval_start_utc=start,
        interval_end_utc=start + SCED_INTERVAL,
        lmp_usd_mwh=interval.lmp_usd_mwh,
        spike_probability=interval.spike_probability,
        curtailment_triggered=curtailed > 0,
        load_served_mw=grid_mw,
        curtailed_mw=curtailed_mw,
        bess_discharge_mw=discharge_mw,
        bess_soc_mwh_end=float(next_soc.quantize(_MW, rounding=ROUND_HALF_UP)),
        energy_cost_usd=energy,
        downtime_cost_usd=penalty,
    )
    return row, next_soc, next_hours


def _allocate_shed(hours_used: dict[str, Decimal], profile: FacilityProfile) -> tuple[Decimal, dict[str, Decimal]]:
    """Shed every discretionary asset that can still run for this 5-minute step."""

    updated = dict(hours_used)
    shed = Decimal("0")
    cap = Decimal(str(profile.power_limits.max_curtailable_load_mw))
    for asset in profile.curtailable_assets:
        remaining = Decimal(str(asset.max_duration_hours)) - updated[asset.name]
        if remaining < _INTERVAL_HOURS:
            continue
        shed += Decimal(str(asset.shed_capacity_mw))
        updated[asset.name] = updated[asset.name] + _INTERVAL_HOURS
    if shed > cap:
        shed = cap
    return shed, updated


def _allocate_discharge(soc: Decimal, profile: FacilityProfile) -> tuple[Decimal, Decimal]:
    """Discharge at nameplate until the reserved state-of-charge floor.

    AC megawatt-hours leave the plant bus. DC megawatt-hours leave the battery
    at ``1 / sqrt(round-trip efficiency)``, so charge and discharge share the
    published 88% round trip.
    """

    battery = profile.battery_storage
    efficiency = Decimal(str(battery.roundtrip_efficiency)).sqrt()
    rated_energy = Decimal(str(battery.rated_energy_mwh))
    min_soc = rated_energy * Decimal(str(battery.min_soc_fraction))
    available_dc = soc - min_soc
    if available_dc <= 0 or efficiency <= 0:
        return Decimal("0"), soc
    available_ac = available_dc * efficiency
    power_cap = Decimal(str(battery.max_discharge_power_mw))
    discharge = min(power_cap, available_ac / _INTERVAL_HOURS).quantize(_MW, rounding=ROUND_HALF_UP)
    if discharge <= 0:
        return Decimal("0"), soc
    next_soc = soc - (discharge * _INTERVAL_HOURS / efficiency)
    floor = min_soc
    if next_soc < floor:
        next_soc = floor
    return discharge, next_soc


def _with_avoided_energy(outcomes: list[StrategyOutcome]) -> list[StrategyOutcome]:
    baseline = next(item for item in outcomes if item.strategy is DispatchStrategy.NO_INTERVENTION)
    updated: list[StrategyOutcome] = []
    for item in outcomes:
        avoided = (
            Decimal("0")
            if item.strategy is DispatchStrategy.NO_INTERVENTION
            else baseline.energy_spend_usd - item.energy_spend_usd
        )
        updated.append(item.model_copy(update={"avoided_energy_spend_usd": avoided}))
    return updated


def _savings_matrix(schedule: DispatchSimulationResult) -> list[StrategySavings]:
    baseline = next(item for item in schedule.outcomes if item.strategy is DispatchStrategy.NO_INTERVENTION)
    rows: list[StrategySavings] = []
    for outcome in schedule.outcomes:
        net = baseline.total_cost_usd - outcome.total_cost_usd
        mitigation = baseline.four_cp_tariff_risk_usd - outcome.four_cp_tariff_risk_usd
        if outcome.strategy is DispatchStrategy.NO_INTERVENTION:
            net = Decimal("0")
            mitigation = Decimal("0")
        rows.append(
            StrategySavings(
                strategy=outcome.strategy,
                energy_spend_usd=outcome.energy_spend_usd,
                downtime_penalty_usd=outcome.downtime_cost_usd,
                total_cost_usd=outcome.total_cost_usd,
                avoided_energy_spend_usd=outcome.avoided_energy_spend_usd,
                net_savings_vs_baseline_usd=net,
                four_cp_tariff_risk_usd=outcome.four_cp_tariff_risk_usd,
                four_cp_risk_mitigation_usd=mitigation,
            )
        )
    return rows


def _ordered_intervals(intervals: list[DispatchIntervalInput]) -> list[DispatchIntervalInput]:
    ordered = sorted(intervals, key=lambda item: item.interval_start_utc)
    for earlier, later in zip(ordered, ordered[1:]):
        expected = earlier.interval_start_utc.astimezone(timezone.utc) + SCED_INTERVAL
        actual = later.interval_start_utc.astimezone(timezone.utc)
        if actual != expected:
            raise ValueError(
                "dispatch intervals must be contiguous 5-minute steps; "
                f"{earlier.interval_start_utc.isoformat()} is not followed by {expected.isoformat()}"
            )
    return ordered


def _initial_soc(requested: float | None, profile: FacilityProfile) -> Decimal:
    battery = profile.battery_storage
    rated = Decimal(str(battery.rated_energy_mwh))
    minimum = rated * Decimal(str(battery.min_soc_fraction))
    maximum = rated * Decimal(str(battery.max_soc_fraction))
    soc = Decimal(str(requested)) if requested is not None else maximum
    if soc < minimum or soc > maximum:
        raise ValueError(f"initial_soc_mwh must sit between {minimum} and {maximum} MWh")
    return soc


def _integrated_mw(values) -> Decimal:
    return sum((Decimal(str(value)) for value in values), Decimal("0")) * _INTERVAL_HOURS


def _money(value: Decimal) -> Decimal:
    return value.quantize(_CENTS, rounding=ROUND_HALF_UP)


def _strategy_rank(strategy: DispatchStrategy) -> int:
    order = (
        DispatchStrategy.NO_INTERVENTION,
        DispatchStrategy.AI_TRIGGERED_CURTAILMENT,
        DispatchStrategy.BESS_PLUS_CURTAILMENT,
    )
    return order.index(strategy)
