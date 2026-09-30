"""Extract 5-minute ERCOT settlement telemetry for HB_HOUSTON and HB_WEST.

The live path reads three public sources and joins them on the SCED interval:

* EMIL NP6-788-CD (MIS report type 12300): ``SCEDTimestamp``, ``RepeatedHourFlag``,
  ``SettlementPoint``, ``LMP``. One zip is one SCED run, normally every five minutes.
* ERCOT dashboards: physical responsive capability, actual system demand, and the
  five-minute fuel mix used for wind and solar.
* NWS observations for the Houston Ship Channel site in ``config/facility_profile.yaml``.

A blocked, empty, or schema-drifted response raises ``TelemetrySourceError``.
``fetch_telemetry`` then returns simulator rows with the same Pydantic contract
and ``authoritative=False``. Those rows are for pipeline continuity only.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import io
import json
import logging
import math
import random
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.api.schemas import RawErcotTelemetry, ScedLmpRow, load_facility_profile

logger = logging.getLogger("ercot.ingestion.fetch")

LMP_EMIL_ID = "NP6-788-CD"
LMP_REPORT_TYPE_ID = "12300"
HUBS = ("HB_HOUSTON", "HB_WEST")
MAX_INTERVALS = 288

DOC_LIST_URL = "https://www.ercot.com/misapp/servlets/IceDocListJsonWS"
DOC_DOWNLOAD_URL = "https://www.ercot.com/misdownload/servlets/mirDownload"
PRC_URL = "https://www.ercot.com/api/1/services/read/dashboards/daily-prc.json"
SUPPLY_DEMAND_URL = "https://www.ercot.com/api/1/services/read/dashboards/supply-demand.json"
FUEL_MIX_URL = "https://www.ercot.com/api/1/services/read/dashboards/fuel-mix.json"

USER_AGENT = "ERCOT-Lakehouse/1.0 (Harris County olefins telemetry; grid-risk)"
NWS_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}
ERCOT_TIMESTAMP = "%Y-%m-%d %H:%M:%S%z"
SCED_TIMESTAMP = "%m/%d/%Y %H:%M:%S"

CHICAGO = ZoneInfo("America/Chicago")
UTC = timezone.utc
SCED_STEP = timedelta(minutes=5)
PRC_TOLERANCE = timedelta(minutes=5)
# Demand and fuel mix are labeled on the five-minute mark. A full step of slack
# would attach the previous interval's megawatts to the current SCED run.
LOAD_TOLERANCE = timedelta(seconds=90)
WEATHER_TOLERANCE = timedelta(hours=2)
# Newest SCED zips sometimes publish before the matching fuel-mix point.
EXTRA_SCED_DOCUMENTS = 3

_SESSION_TIMEOUT = (5.0, 45.0)


class TelemetrySourceError(RuntimeError):
    """The live ERCOT or NWS feed could not produce a complete interval."""


@dataclass(frozen=True)
class _DocumentRef:
    doc_id: str
    published_at: datetime
    friendly_name: str


@dataclass(frozen=True)
class _HubQuote:
    row: ScedLmpRow
    document_id: str
    published_at: datetime


@dataclass(frozen=True)
class _TimedValue:
    at: datetime
    value: float


@dataclass(frozen=True)
class _WeatherObservation:
    observed_at: datetime
    station_id: str
    temperature_f: float
    relative_humidity_pct: float
    heat_index_f: float


def build_session() -> requests.Session:
    """HTTP session with bounded retries for the public ERCOT and NWS endpoints."""

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=8)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return session


def fetch_telemetry(
    *,
    intervals: int = 1,
    allow_simulation_fallback: bool = True,
    simulate: bool = False,
    seed: int | None = None,
    session: requests.Session | None = None,
) -> list[RawErcotTelemetry]:
    """Return Houston and West hub rows for the latest ``intervals`` SCED runs.

    ``intervals`` counts five-minute timestamps, not hub rows. Each timestamp
    produces one HB_HOUSTON row and one HB_WEST row.
    """

    _validate_intervals(intervals)
    if simulate:
        return simulate_telemetry(
            intervals=intervals,
            seed=seed,
            reason="operator requested the fallback simulator",
        )
    profile = load_facility_profile()
    owns_session = session is None
    http = session or build_session()
    try:
        return _fetch_live(
            intervals=intervals,
            session=http,
            latitude=profile.weather_station.latitude,
            longitude=profile.weather_station.longitude,
        )
    except TelemetrySourceError as exc:
        logger.warning("Live ERCOT/NWS telemetry unavailable: %s", exc)
        if not allow_simulation_fallback:
            raise
        return simulate_telemetry(intervals=intervals, seed=seed, reason=str(exc))
    finally:
        if owns_session:
            http.close()


def simulate_telemetry(
    *,
    intervals: int,
    seed: int | None = None,
    reason: str,
) -> list[RawErcotTelemetry]:
    """Build a non-authoritative series with the bronze telemetry contract.

    The series is an autocorrelated Houston summer-shoulder day: system demand,
    a solar bell, a mean-reverting wind plant, PRC that tightens with net load,
    and a Houston-minus-West congestion spread. It does not reproduce a specific
    historical operating day.
    """

    _validate_intervals(intervals)
    cleaned_reason = reason.strip()[:4000] or "live telemetry source failed"
    rng = random.Random(_resolve_seed(seed))
    ingested_at = datetime.now(UTC).replace(microsecond=0)
    interval_starts = _simulator_interval_starts(intervals)
    demand_shock = _ar1(rng, intervals, phi=0.92, sigma=180.0)
    wind_shock = _ar1(rng, intervals, phi=0.97, sigma=140.0)
    prc_shock = _ar1(rng, intervals, phi=0.95, sigma=120.0)
    price_shock = _ar1(rng, intervals, phi=0.85, sigma=0.8)
    spread_shock = _ar1(rng, intervals, phi=0.80, sigma=0.6)

    records: list[RawErcotTelemetry] = []
    for index, interval_start in enumerate(interval_starts):
        hour = _hour_fraction(interval_start)
        diurnal = math.sin(2.0 * math.pi * (hour - 16.0) / 24.0)
        shoulder = math.sin(4.0 * math.pi * (hour - 8.0) / 24.0)
        demand = 62_000.0 + 7_000.0 * diurnal + 1_800.0 * shoulder + demand_shock[index]
        demand = min(85_000.0, max(40_000.0, demand))
        if 6.5 < hour < 18.5:
            solar_argument = (hour - 6.5) / 12.0
            solar = math.sin(math.pi * solar_argument) * 26_000.0
        else:
            solar = 0.0
        solar = min(40_000.0, max(0.0, solar + rng.gauss(0.0, 40.0)))
        wind = min(28_000.0, max(400.0, 9_000.0 + wind_shock[index]))
        net_load = demand - wind - solar
        reserve = 16_000.0 - 0.25 * (net_load - 40_000.0) + prc_shock[index]
        reserve = min(30_000.0, max(800.0, reserve))
        online_capacity = min(160_000.0, demand + max(8_000.0, reserve * 0.45))
        tightness = max(0.0, (7_000.0 - reserve) / 7_000.0)
        heat_index = 84.0 + 8.0 * math.sin(2.0 * math.pi * (hour - 15.0) / 24.0)
        heat_index = min(110.0, max(70.0, heat_index))
        humidity = 72.0 - 12.0 * math.sin(2.0 * math.pi * (hour - 15.0) / 24.0)
        humidity = min(95.0, max(35.0, humidity))
        temperature = min(105.0, max(60.0, heat_index - 5.0))
        houston_lmp = (
            22.0
            + 0.00015 * max(net_load, 0.0)
            + 35.0 * tightness
            + 80.0 * tightness * tightness
            + max(0.0, (heat_index - 90.0) * 0.4)
            + price_shock[index]
        )
        houston_lmp = min(5_000.0, max(-50.0, houston_lmp))
        spread = 1.5 + 0.8 * tightness + 0.35 * spread_shock[index]
        spread = min(40.0, max(-15.0, spread))
        west_lmp = min(5_000.0, max(-50.0, houston_lmp - spread))
        if reserve < 2_500.0:
            eea_level = 2
            grid_state = "eea2"
        elif reserve < 3_000.0:
            eea_level = 1
            grid_state = "eea1"
        else:
            eea_level = 0
            grid_state = "normal"
        sced_local = interval_start + timedelta(seconds=15)
        observed_at = interval_start.astimezone(UTC)
        for settlement_point, lmp in (("HB_HOUSTON", houston_lmp), ("HB_WEST", west_lmp)):
            records.append(
                RawErcotTelemetry(
                    sced_timestamp_utc=_as_utc(sced_local),
                    interval_start_utc=_as_utc(interval_start),
                    interval_end_utc=_as_utc(interval_start + SCED_STEP),
                    repeated_hour_flag="Y" if sced_local.fold == 1 else "N",
                    settlement_point=settlement_point,
                    settlement_point_type="HU",
                    lmp_usd_mwh=round(lmp, 2),
                    operating_reserve_mw=round(reserve, 3),
                    system_demand_mw=round(demand, 3),
                    online_capacity_mw=round(online_capacity, 3),
                    wind_generation_mw=round(wind, 3),
                    solar_generation_mw=round(solar, 3),
                    net_load_mw=round(round(demand, 3) - round(wind, 3) - round(solar, 3), 3),
                    eea_level=eea_level,
                    grid_state=grid_state,
                    heat_index_f=round(heat_index, 3),
                    temperature_f=round(temperature, 3),
                    relative_humidity_pct=round(humidity, 3),
                    weather_station_id="SIM-KT41",
                    weather_observed_at_utc=observed_at,
                    emil_id=None,
                    report_type_id=None,
                    document_id=None,
                    published_at_utc=None,
                    source="simulator",
                    authoritative=False,
                    fallback_reason=cleaned_reason,
                    ingested_at_utc=ingested_at,
                )
            )
    return records


def _fetch_live(
    *,
    intervals: int,
    session: requests.Session,
    latitude: float,
    longitude: float,
) -> list[RawErcotTelemetry]:
    try:
        return _assemble_live(
            intervals=intervals,
            session=session,
            latitude=latitude,
            longitude=longitude,
        )
    except TelemetrySourceError:
        raise
    except (requests.RequestException, zipfile.BadZipFile, csv.Error, KeyError, TypeError, ValueError) as exc:
        raise TelemetrySourceError(f"live telemetry parse or transport failed: {exc}") from exc


def _assemble_live(
    *,
    intervals: int,
    session: requests.Session,
    latitude: float,
    longitude: float,
) -> list[RawErcotTelemetry]:
    documents = _list_lmp_documents(session, limit=min(MAX_INTERVALS, intervals + EXTRA_SCED_DOCUMENTS))
    quotes = _download_hub_quotes(documents)
    grouped = _group_quotes(quotes)
    if not grouped:
        raise TelemetrySourceError("NP6-788-CD returned no complete HB_HOUSTON / HB_WEST intervals")

    ordered = sorted(grouped)
    window_start = ordered[0] - WEATHER_TOLERANCE
    window_end = ordered[-1] + timedelta(minutes=10)
    reserves, eea_level, grid_state = _fetch_prc(session)
    demand, capacity = _fetch_supply_demand(session)
    wind, solar = _fetch_fuel_mix(session)
    weather = _fetch_weather(session, latitude, longitude, window_start, window_end)

    ingested_at = datetime.now(UTC).replace(microsecond=0)
    newest = ordered[-1]
    kept: list[tuple[datetime, _HubQuote, _HubQuote]] = []
    failures: list[str] = []
    for sced_timestamp in ordered:
        try:
            _asof(reserves, sced_timestamp, PRC_TOLERANCE)
            interval_start = _floor_five_minutes(sced_timestamp.astimezone(CHICAGO))
            _asof(demand, interval_start, LOAD_TOLERANCE)
            _asof(capacity, interval_start, LOAD_TOLERANCE)
            _asof(wind, interval_start, LOAD_TOLERANCE)
            _asof(solar, interval_start, LOAD_TOLERANCE)
            _asof_weather(weather, sced_timestamp, WEATHER_TOLERANCE)
        except TelemetrySourceError as exc:
            failures.append(f"{sced_timestamp.isoformat()}: {exc}")
            continue
        houston, west = grouped[sced_timestamp]
        kept.append((sced_timestamp, houston, west))

    if not kept:
        detail = "; ".join(failures[:3])
        raise TelemetrySourceError(f"no SCED interval could be joined to reserves, load, and weather ({detail})")
    if len(kept) > intervals:
        kept = kept[-intervals:]
    if failures:
        logger.warning(
            "Dropped %s SCED intervals that were not on the same 5-minute demand and fuel-mix stamp",
            len(failures),
        )
    if len(kept) < intervals:
        logger.warning(
            "Returning %s of %s requested SCED intervals; newer runs were not aligned yet",
            len(kept),
            intervals,
        )

    records: list[RawErcotTelemetry] = []
    for sced_timestamp, houston, west in kept:
        interval_start_local = _floor_five_minutes(sced_timestamp.astimezone(CHICAGO))
        interval_start = _as_utc(interval_start_local)
        interval_end = _as_utc(interval_start_local + SCED_STEP)
        reserve_mw = _asof(reserves, sced_timestamp, PRC_TOLERANCE)
        demand_mw = _asof(demand, interval_start_local, LOAD_TOLERANCE)
        capacity_mw = _asof(capacity, interval_start_local, LOAD_TOLERANCE)
        wind_mw = max(0.0, _asof(wind, interval_start_local, LOAD_TOLERANCE))
        solar_mw = max(0.0, _asof(solar, interval_start_local, LOAD_TOLERANCE))
        observation = _asof_weather(weather, sced_timestamp, WEATHER_TOLERANCE)
        attach_condition = sced_timestamp == newest
        for quote in (houston, west):
            records.append(
                RawErcotTelemetry(
                    sced_timestamp_utc=_as_utc(sced_timestamp),
                    interval_start_utc=interval_start,
                    interval_end_utc=interval_end,
                    repeated_hour_flag=quote.row.repeated_hour_flag,
                    settlement_point=quote.row.settlement_point,
                    settlement_point_type="HU",
                    lmp_usd_mwh=quote.row.lmp_usd_mwh,
                    operating_reserve_mw=round(reserve_mw, 3),
                    system_demand_mw=round(demand_mw, 3),
                    online_capacity_mw=round(capacity_mw, 3),
                wind_generation_mw=round(wind_mw, 3),
                solar_generation_mw=round(solar_mw, 3),
                net_load_mw=round(round(demand_mw, 3) - round(wind_mw, 3) - round(solar_mw, 3), 3),
                    eea_level=eea_level if attach_condition else None,
                    grid_state=grid_state if attach_condition else None,
                    heat_index_f=round(observation.heat_index_f, 3),
                    temperature_f=round(observation.temperature_f, 3),
                    relative_humidity_pct=round(observation.relative_humidity_pct, 3),
                    weather_station_id=observation.station_id,
                    weather_observed_at_utc=_as_utc(observation.observed_at),
                    emil_id=LMP_EMIL_ID,
                    report_type_id=LMP_REPORT_TYPE_ID,
                    document_id=quote.document_id,
                    published_at_utc=_as_utc(quote.published_at),
                    source="ercot_live",
                    authoritative=True,
                    fallback_reason=None,
                    ingested_at_utc=ingested_at,
                )
            )
    records.sort(key=lambda row: (row.sced_timestamp_utc, row.settlement_point))
    logger.info(
        "Assembled %s live telemetry rows across %s SCED intervals",
        len(records),
        len(kept),
    )
    return records


def _list_lmp_documents(session: requests.Session, *, limit: int) -> list[_DocumentRef]:
    payload = _get_json(session, DOC_LIST_URL, params={"reportTypeId": LMP_REPORT_TYPE_ID})
    try:
        listed = payload["ListDocsByRptTypeRes"]["DocumentList"]
    except (KeyError, TypeError) as exc:
        raise TelemetrySourceError("MIS document list did not contain ListDocsByRptTypeRes.DocumentList") from exc
    documents: list[_DocumentRef] = []
    for item in listed:
        document = item.get("Document") if isinstance(item, dict) else None
        if not isinstance(document, dict):
            continue
        friendly = str(document.get("FriendlyName") or "")
        extension = str(document.get("Extension") or "").lower()
        if extension != "zip" or "_csv" not in friendly.lower():
            continue
        doc_id = str(document.get("DocID") or "").strip()
        published_raw = str(document.get("PublishDate") or "").strip()
        if not doc_id or not published_raw:
            continue
        published_at = datetime.fromisoformat(published_raw)
        documents.append(_DocumentRef(doc_id=doc_id, published_at=published_at, friendly_name=friendly))
    documents.sort(key=lambda doc: doc.published_at, reverse=True)
    selected = documents[:limit]
    if not selected:
        raise TelemetrySourceError(f"No CSV publications found for report type {LMP_REPORT_TYPE_ID}")
    logger.info(
        "Selected %s NP6-788-CD CSV documents ending at %s",
        len(selected),
        selected[0].friendly_name,
    )
    return selected


def _download_hub_quotes(documents: list[_DocumentRef]) -> list[_HubQuote]:
    quotes: list[_HubQuote] = []
    workers = min(6, len(documents))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_and_parse, document): document for document in documents}
        for future in as_completed(futures):
            quotes.extend(future.result())
    return quotes


def _download_and_parse(document: _DocumentRef) -> list[_HubQuote]:
    with build_session() as session:
        payload = _get_bytes(session, DOC_DOWNLOAD_URL, params={"doclookupId": document.doc_id})
    if len(payload) < 22 or payload[:2] != b"PK":
        raise TelemetrySourceError(
            f"document {document.doc_id} is not a zip archive ({len(payload)} bytes)"
        )
    return _parse_lmp_zip(payload, document)


def _parse_lmp_zip(payload: bytes, document: _DocumentRef) -> list[_HubQuote]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise TelemetrySourceError(
                f"document {document.doc_id} contains {len(csv_names)} CSV members, expected 1"
            )
        with archive.open(csv_names[0]) as handle:
            text = io.TextIOWrapper(handle, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(text)
            found: dict[str, ScedLmpRow] = {}
            for raw in reader:
                normalized = {
                    (key or "").strip().lower(): (value or "").strip()
                    for key, value in raw.items()
                }
                settlement_point = normalized.get("settlementpoint", "")
                if settlement_point not in HUBS:
                    continue
                flag = normalized.get("repeatedhourflag", "")
                if flag not in {"Y", "N"}:
                    raise TelemetrySourceError(
                        f"document {document.doc_id} has RepeatedHourFlag {flag!r} for {settlement_point}"
                    )
                try:
                    lmp = float(normalized["lmp"])
                    naive = datetime.strptime(normalized["scedtimestamp"], SCED_TIMESTAMP)
                except (KeyError, TypeError, ValueError) as exc:
                    raise TelemetrySourceError(
                        f"document {document.doc_id} has an unreadable {settlement_point} LMP row"
                    ) from exc
                aware = naive.replace(tzinfo=CHICAGO, fold=1 if flag == "Y" else 0)
                found[settlement_point] = ScedLmpRow(
                    sced_timestamp_ct=aware,
                    repeated_hour_flag=flag,
                    settlement_point=settlement_point,  # type: ignore[arg-type]
                    lmp_usd_mwh=lmp,
                )
    missing = [hub for hub in HUBS if hub not in found]
    if missing:
        raise TelemetrySourceError(f"document {document.doc_id} is missing {', '.join(missing)}")
    houston_ts = found["HB_HOUSTON"].sced_timestamp_ct
    west_ts = found["HB_WEST"].sced_timestamp_ct
    if houston_ts != west_ts:
        raise TelemetrySourceError(f"document {document.doc_id} has different hub SCED timestamps")
    return [
        _HubQuote(row=found[hub], document_id=document.doc_id, published_at=document.published_at)
        for hub in HUBS
    ]


def _group_quotes(quotes: list[_HubQuote]) -> dict[datetime, tuple[_HubQuote, _HubQuote]]:
    latest: dict[tuple[str, datetime], _HubQuote] = {}
    for quote in quotes:
        sced_utc = _as_utc(quote.row.sced_timestamp_ct)
        key = (quote.row.settlement_point, sced_utc)
        current = latest.get(key)
        if current is None or quote.published_at >= current.published_at:
            latest[key] = quote
    grouped: dict[datetime, dict[str, _HubQuote]] = {}
    for (settlement_point, sced_utc), quote in latest.items():
        grouped.setdefault(sced_utc, {})[settlement_point] = quote
    complete: dict[datetime, tuple[_HubQuote, _HubQuote]] = {}
    for sced_utc, pair in grouped.items():
        if "HB_HOUSTON" in pair and "HB_WEST" in pair:
            complete[sced_utc] = (pair["HB_HOUSTON"], pair["HB_WEST"])
    return complete


def _fetch_prc(session: requests.Session) -> tuple[list[_TimedValue], int, str]:
    payload = _get_json(session, PRC_URL)
    rows = payload.get("data")
    if not isinstance(rows, list) or not rows:
        raise TelemetrySourceError("daily-prc.json did not contain a PRC series")
    series: list[_TimedValue] = []
    for row in rows:
        if not isinstance(row, dict) or "timestamp" not in row or "prc" not in row:
            continue
        series.append(
            _TimedValue(
                at=datetime.strptime(str(row["timestamp"]), ERCOT_TIMESTAMP),
                value=float(row["prc"]),
            )
        )
    if not series:
        raise TelemetrySourceError("daily-prc.json series had no parseable points")
    series.sort(key=lambda item: item.at)
    condition = payload.get("current_condition")
    if not isinstance(condition, dict):
        raise TelemetrySourceError("daily-prc.json did not contain current_condition")
    try:
        eea_level = int(condition["eea_level"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TelemetrySourceError("current_condition.eea_level is missing") from exc
    if eea_level < 0 or eea_level > 3:
        raise TelemetrySourceError(f"eea_level {eea_level} is outside 0..3")
    state = str(condition.get("state") or condition.get("title") or "").strip().lower()
    if not state:
        raise TelemetrySourceError("current_condition has no grid state")
    return series, eea_level, state[:64]


def _fetch_supply_demand(session: requests.Session) -> tuple[list[_TimedValue], list[_TimedValue]]:
    payload = _get_json(session, SUPPLY_DEMAND_URL)
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise TelemetrySourceError("supply-demand.json did not contain data")
    demand: list[_TimedValue] = []
    capacity: list[_TimedValue] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if int(row.get("forecast", 1)) != 0:
            continue
        stamp = datetime.strptime(str(row["timestamp"]), ERCOT_TIMESTAMP)
        demand.append(_TimedValue(at=stamp, value=float(row["demand"])))
        capacity.append(_TimedValue(at=stamp, value=float(row["capacity"])))
    if not demand:
        raise TelemetrySourceError("supply-demand.json has no actual (forecast=0) intervals")
    demand.sort(key=lambda item: item.at)
    capacity.sort(key=lambda item: item.at)
    return demand, capacity


def _fetch_fuel_mix(session: requests.Session) -> tuple[list[_TimedValue], list[_TimedValue]]:
    payload = _get_json(session, FUEL_MIX_URL)
    days = payload.get("data")
    if not isinstance(days, dict) or not days:
        raise TelemetrySourceError("fuel-mix.json did not contain a data object")
    wind: list[_TimedValue] = []
    solar: list[_TimedValue] = []
    for day in days.values():
        if not isinstance(day, dict):
            continue
        for stamp, fuels in day.items():
            if not isinstance(fuels, dict):
                continue
            try:
                at = datetime.strptime(str(stamp), ERCOT_TIMESTAMP)
                wind_mw = float(fuels["Wind"]["gen"])
                solar_mw = float(fuels["Solar"]["gen"])
            except (KeyError, TypeError, ValueError):
                continue
            wind.append(_TimedValue(at=at, value=max(0.0, wind_mw)))
            solar.append(_TimedValue(at=at, value=max(0.0, solar_mw)))
    if not wind or not solar:
        raise TelemetrySourceError("fuel-mix.json has no five-minute wind and solar generation")
    wind.sort(key=lambda item: item.at)
    solar.sort(key=lambda item: item.at)
    return wind, solar


def _fetch_weather(
    session: requests.Session,
    latitude: float,
    longitude: float,
    start: datetime,
    end: datetime,
) -> list[_WeatherObservation]:
    points_url = f"https://api.weather.gov/points/{latitude:.4f},{longitude:.4f}"
    points = _get_json(session, points_url, headers=NWS_HEADERS)
    try:
        stations_url = points["properties"]["observationStations"]
    except (KeyError, TypeError) as exc:
        raise TelemetrySourceError("NWS points response has no observationStations link") from exc
    stations = _get_json(session, stations_url, headers=NWS_HEADERS)
    features = stations.get("features") if isinstance(stations, dict) else None
    if not isinstance(features, list) or not features:
        raise TelemetrySourceError("NWS returned no stations for the Houston Ship Channel point")

    errors: list[str] = []
    for feature in features[:3]:
        properties = feature.get("properties") if isinstance(feature, dict) else None
        station_id = str((properties or {}).get("stationIdentifier") or "").strip()
        if not station_id:
            continue
        try:
            observations = _fetch_station_observations(session, station_id, start, end)
        except TelemetrySourceError as exc:
            errors.append(f"{station_id}: {exc}")
            continue
        if observations:
            logger.info("Using NWS station %s (%s observations in window)", station_id, len(observations))
            return observations
        errors.append(f"{station_id}: no temperature observations in window")
    detail = "; ".join(errors) or "no station identifiers"
    raise TelemetrySourceError(f"Harris County heat index unavailable ({detail})")


def _fetch_station_observations(
    session: requests.Session,
    station_id: str,
    start: datetime,
    end: datetime,
) -> list[_WeatherObservation]:
    url = f"https://api.weather.gov/stations/{station_id}/observations"
    params = {
        "start": start.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": "500",
    }
    payload = _get_json(session, url, params=params, headers=NWS_HEADERS)
    observations = _parse_observation_collection(payload, station_id)
    next_url = payload.get("pagination", {}).get("next") if isinstance(payload, dict) else None
    pages = 0
    while next_url and pages < 3:
        payload = _get_json(session, str(next_url), headers=NWS_HEADERS)
        observations.extend(_parse_observation_collection(payload, station_id))
        next_url = payload.get("pagination", {}).get("next") if isinstance(payload, dict) else None
        pages += 1
    if not observations:
        latest = _get_json(session, f"{url}/latest", headers=NWS_HEADERS)
        observations = _parse_observation_collection({"features": [latest]}, station_id)
    observations.sort(key=lambda item: item.observed_at)
    return observations


def _parse_observation_collection(payload: dict, station_id: str) -> list[_WeatherObservation]:
    features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(features, list):
        return []
    parsed: list[_WeatherObservation] = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        observation = _parse_observation_feature(feature, station_id)
        if observation is not None:
            parsed.append(observation)
    return parsed


def _parse_observation_feature(feature: dict, station_id: str) -> _WeatherObservation | None:
    properties = feature.get("properties")
    if not isinstance(properties, dict) or "timestamp" not in properties:
        return None
    temperature_c = _quantity(properties.get("temperature"))
    humidity = _quantity(properties.get("relativeHumidity"))
    if temperature_c is None or humidity is None:
        return None
    observed_at = datetime.fromisoformat(str(properties["timestamp"]))
    temperature_f = _celsius_to_fahrenheit(temperature_c)
    humidity = min(100.0, max(0.0, humidity))
    heat_index_c = _quantity(properties.get("heatIndex"))
    if heat_index_c is None:
        heat_index_f = heat_index_fahrenheit(temperature_f, humidity)
    else:
        heat_index_f = _celsius_to_fahrenheit(heat_index_c)
    return _WeatherObservation(
        observed_at=observed_at,
        station_id=station_id,
        temperature_f=temperature_f,
        relative_humidity_pct=humidity,
        heat_index_f=heat_index_f,
    )


def heat_index_fahrenheit(temperature_f: float, relative_humidity_pct: float) -> float:
    """Rothfusz regression used by the NWS when the reported heat index is null.

    Below 80 F the heat index is the dry-bulb temperature. The adjustment terms
    match the NWS implementation for low humidity and for muggy cool air.
    """

    if temperature_f < 80.0:
        return temperature_f
    temperature = temperature_f
    humidity = relative_humidity_pct
    heat_index = (
        -42.379
        + 2.04901523 * temperature
        + 10.14333127 * humidity
        - 0.22475541 * temperature * humidity
        - 0.00683783 * temperature * temperature
        - 0.05481717 * humidity * humidity
        + 0.00122874 * temperature * temperature * humidity
        + 0.00085282 * temperature * humidity * humidity
        - 0.00000199 * temperature * temperature * humidity * humidity
    )
    if humidity < 13.0 and 80.0 <= temperature <= 112.0:
        heat_index -= ((13.0 - humidity) / 4.0) * math.sqrt((17.0 - abs(temperature - 95.0)) / 17.0)
    elif humidity > 85.0 and 80.0 <= temperature <= 87.0:
        heat_index += ((humidity - 85.0) / 10.0) * ((87.0 - temperature) / 5.0)
    return heat_index


def _asof(series: list[_TimedValue], moment: datetime, tolerance: timedelta) -> float:
    if not series:
        raise TelemetrySourceError("as-of join series is empty")
    index = bisect.bisect_right(series, moment, key=lambda item: item.at) - 1
    if index < 0:
        raise TelemetrySourceError(f"no observation at or before {moment.isoformat()}")
    age = moment - series[index].at
    if age > tolerance:
        raise TelemetrySourceError(
            f"observation at {series[index].at.isoformat()} is {age} before {moment.isoformat()}"
        )
    return series[index].value


def _asof_weather(
    series: list[_WeatherObservation],
    moment: datetime,
    tolerance: timedelta,
) -> _WeatherObservation:
    if not series:
        raise TelemetrySourceError("weather observation series is empty")
    index = bisect.bisect_right(series, moment, key=lambda item: item.observed_at) - 1
    if index < 0:
        raise TelemetrySourceError(f"no weather observation at or before {moment.isoformat()}")
    chosen = series[index]
    age = moment - chosen.observed_at
    if age > tolerance:
        raise TelemetrySourceError(
            f"weather observation {chosen.station_id} at {chosen.observed_at.isoformat()} is stale by {age}"
        )
    return chosen


def _get_json(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> dict:
    response = session.get(url, params=params, headers=headers, timeout=_SESSION_TIMEOUT)
    _raise_for_blocked(response, url)
    text = response.text.lstrip()
    if not text.startswith(("{", "[")):
        raise TelemetrySourceError(f"{url} returned a non-JSON body ({response.status_code})")
    try:
        payload = response.json()
    except ValueError as exc:
        raise TelemetrySourceError(f"{url} returned malformed JSON") from exc
    if not isinstance(payload, dict):
        raise TelemetrySourceError(f"{url} returned a JSON {type(payload).__name__}, expected an object")
    return payload


def _get_bytes(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, str] | None = None,
) -> bytes:
    response = session.get(url, params=params, timeout=(5.0, 60.0))
    _raise_for_blocked(response, url)
    return response.content


def _raise_for_blocked(response: requests.Response, url: str) -> None:
    if response.status_code in {401, 403, 451}:
        raise TelemetrySourceError(f"{url} returned {response.status_code}; the public endpoint blocked the client")
    if response.status_code >= 400:
        snippet = response.text[:180].replace("\n", " ")
        raise TelemetrySourceError(f"{url} returned {response.status_code}: {snippet}")


def _floor_five_minutes(moment: datetime) -> datetime:
    local = moment.astimezone(CHICAGO)
    minute = local.minute - (local.minute % 5)
    return local.replace(minute=minute, second=0, microsecond=0, fold=local.fold)


def _as_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        raise TelemetrySourceError("refusing to convert a naive timestamp")
    return moment.astimezone(UTC).replace(microsecond=0)


def _simulator_interval_starts(intervals: int) -> list[datetime]:
    now_local = datetime.now(CHICAGO).replace(second=0, microsecond=0)
    latest = now_local.replace(minute=now_local.minute - (now_local.minute % 5))
    oldest = latest - SCED_STEP * (intervals - 1)
    return [oldest + SCED_STEP * index for index in range(intervals)]


def _hour_fraction(moment: datetime) -> float:
    local = moment.astimezone(CHICAGO)
    return local.hour + local.minute / 60.0 + local.second / 3600.0


def _ar1(rng: random.Random, count: int, *, phi: float, sigma: float) -> list[float]:
    values = [rng.gauss(0.0, sigma)]
    for _ in range(1, count):
        values.append(phi * values[-1] + rng.gauss(0.0, sigma))
    return values


def _resolve_seed(seed: int | None) -> int:
    if seed is not None:
        return seed
    return int(datetime.now(UTC).timestamp()) // 300


def _quantity(block: object) -> float | None:
    if not isinstance(block, dict):
        return None
    value = block.get("value")
    if value is None:
        return None
    return float(value)


def _celsius_to_fahrenheit(celsius: float) -> float:
    return celsius * 9.0 / 5.0 + 32.0


def _validate_intervals(intervals: int) -> None:
    if intervals < 1 or intervals > MAX_INTERVALS:
        raise ValueError(f"intervals must be between 1 and {MAX_INTERVALS}")


def _summary(records: list[RawErcotTelemetry]) -> str:
    houston = [row for row in records if row.settlement_point == "HB_HOUSTON"]
    latest = houston[-1]
    return (
        f"source={latest.source} intervals={len(houston)} "
        f"sced={latest.sced_timestamp_utc.isoformat()} "
        f"HB_HOUSTON={latest.lmp_usd_mwh:.2f} "
        f"reserve_mw={latest.operating_reserve_mw:.1f} "
        f"net_load_mw={latest.net_load_mw:.1f} "
        f"heat_index_f={latest.heat_index_f:.1f} "
        f"station={latest.weather_station_id}"
    )


def main(argv: list[str] | None = None) -> int:
    """Write the latest telemetry batch as JSON."""

    parser = argparse.ArgumentParser(description="Extract HB_HOUSTON and HB_WEST 5-minute SCED telemetry.")
    parser.add_argument("--intervals", type=int, default=1, help="Number of 5-minute SCED intervals.")
    parser.add_argument("--simulate", action="store_true", help="Skip the live endpoints and use the fallback simulator.")
    parser.add_argument("--no-fallback", action="store_true", help="Raise if the live endpoints fail.")
    parser.add_argument("--seed", type=int, default=None, help="Simulator seed. Default is the current 5-minute bucket.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path. Default is stdout.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    records = fetch_telemetry(
        intervals=args.intervals,
        allow_simulation_fallback=not args.no_fallback,
        simulate=args.simulate,
        seed=args.seed,
    )
    logger.info(_summary(records))
    payload = [record.model_dump(mode="json") for record in records]
    text = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
