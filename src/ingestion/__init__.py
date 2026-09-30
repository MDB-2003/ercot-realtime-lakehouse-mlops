"""Telemetry extraction and bronze landing.

Imports are lazy so ``python -m src.ingestion.fetch_ercot`` does not load this
module's public names before the CLI module executes.
"""

from __future__ import annotations

from typing import Any

__all__ = ["TelemetrySourceError", "fetch_telemetry", "simulate_telemetry"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from src.ingestion.fetch_ercot import TelemetrySourceError, fetch_telemetry, simulate_telemetry

        return {
            "TelemetrySourceError": TelemetrySourceError,
            "fetch_telemetry": fetch_telemetry,
            "simulate_telemetry": simulate_telemetry,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
