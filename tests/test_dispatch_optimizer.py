"""Lock the ship-channel dispatch economics on one 5-minute interval."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from decimal import Decimal

from src.api.schemas import DispatchIntervalInput, DispatchSimulationRequest
from src.models.dispatch_optimizer import simulate_dispatch


def _one_interval(price: float, probability: float, threshold: float = 0.5):
    request = DispatchSimulationRequest(
        intervals=[
            DispatchIntervalInput(
                interval_start_utc=datetime(2026, 9, 30, 16, 0, tzinfo=timezone.utc),
                lmp_usd_mwh=price,
                spike_probability=probability,
            )
        ],
        decision_threshold=threshold,
    )
    result = simulate_dispatch(request)
    outcomes = {row.strategy.value: row for row in result.schedule.outcomes}
    savings = {row.strategy.value: row for row in result.savings_matrix}
    return result, outcomes, savings


class DispatchEconomicsTest(unittest.TestCase):
    def test_high_price_prefers_battery_plus_shed(self) -> None:
        result, outcomes, savings = _one_interval(400.0, 0.9)
        baseline = outcomes["no_intervention"]
        curtailment = outcomes["ai_triggered_curtailment"]
        both = outcomes["bess_plus_curtailment"]

        self.assertEqual(baseline.energy_spend_usd, Decimal("1666.67"))
        self.assertEqual(baseline.downtime_cost_usd, Decimal("0"))
        self.assertEqual(baseline.total_cost_usd, Decimal("1666.67"))
        self.assertEqual(baseline.four_cp_tariff_risk_usd, Decimal("2700000.00"))

        self.assertEqual(curtailment.intervals[0].curtailed_mw, 20.0)
        self.assertEqual(curtailment.intervals[0].load_served_mw, 30.0)
        self.assertEqual(curtailment.energy_spend_usd, Decimal("1000.00"))
        self.assertEqual(curtailment.downtime_cost_usd, Decimal("541.67"))
        self.assertEqual(curtailment.total_cost_usd, Decimal("1541.67"))
        self.assertEqual(savings["ai_triggered_curtailment"].four_cp_risk_mitigation_usd, Decimal("1080000.00"))

        self.assertEqual(both.intervals[0].bess_discharge_mw, 10.0)
        self.assertEqual(both.intervals[0].curtailed_mw, 20.0)
        self.assertEqual(both.intervals[0].load_served_mw, 20.0)
        self.assertEqual(both.energy_spend_usd, Decimal("666.67"))
        self.assertEqual(both.downtime_cost_usd, Decimal("541.67"))
        self.assertEqual(both.total_cost_usd, Decimal("1208.34"))
        self.assertEqual(savings["bess_plus_curtailment"].net_savings_vs_baseline_usd, Decimal("458.33"))
        self.assertEqual(savings["bess_plus_curtailment"].four_cp_risk_mitigation_usd, Decimal("1620000.00"))
        self.assertEqual(result.schedule.lowest_cost_strategy.value, "bess_plus_curtailment")
        self.assertIs(result.schedule.llm_authorized_trigger, False)

    def test_threshold_tie_does_not_shed(self) -> None:
        result, outcomes, _savings = _one_interval(400.0, 0.5)
        self.assertTrue(all(row.intervals[0].curtailment_triggered is False for row in outcomes.values()))
        self.assertEqual(result.schedule.lowest_cost_strategy.value, "no_intervention")

    def test_calm_price_keeps_the_baseline(self) -> None:
        result, _outcomes, savings = _one_interval(35.0, 0.95)
        self.assertEqual(result.schedule.lowest_cost_strategy.value, "no_intervention")
        self.assertLess(savings["ai_triggered_curtailment"].net_savings_vs_baseline_usd, 0)


if __name__ == "__main__":
    unittest.main()
