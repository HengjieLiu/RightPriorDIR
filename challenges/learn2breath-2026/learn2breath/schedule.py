"""Submitted five-level pyramid and per-level early stopping for Learn2Breath."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ScheduleConfig:
    """One complete five-stage configuration, including its stop rule."""

    variant_id: str
    full_cps: tuple[int, int, int, int, int]
    downsample: tuple[int, int, int, int, int]
    epoch_caps: tuple[int, int, int, int, int]
    lrs: tuple[float, float, float, float, float]
    early_stop_enabled: bool
    rule: str | None
    check_interval_epochs: int
    window_epochs: int
    minimum_epochs: tuple[int, int, int, int, int]
    configured_patience_epochs: tuple[int, int, int, int, int]
    effective_patience_epochs: tuple[int, int, int, int, int]
    total_relative_improvement_thresholds: tuple[float, float, float, float, float]
    data_relative_improvement_thresholds: tuple[float, float, float, float, float]

    def as_dict(self) -> dict[str, Any]:
        """Return a report-friendly immutable schedule record."""

        return {
            "variant_id": self.variant_id,
            "full_cps": list(self.full_cps),
            "downsample": list(self.downsample),
            "epoch_caps": list(self.epoch_caps),
            "lrs": list(self.lrs),
            "early_stop": {
                "enabled": self.early_stop_enabled,
                "rule": self.rule,
                "check_interval_epochs": self.check_interval_epochs,
                "window_epochs": self.window_epochs,
                "minimum_epochs": list(self.minimum_epochs),
                "configured_patience_epochs": list(self.configured_patience_epochs),
                "effective_patience_epochs": list(self.effective_patience_epochs),
                "total_relative_improvement_thresholds": list(
                    self.total_relative_improvement_thresholds
                ),
                "data_relative_improvement_thresholds": list(
                    self.data_relative_improvement_thresholds
                ),
            },
        }


SCHEDULE = ScheduleConfig(
    variant_id="oasis_bal_decay_ordinal",
    full_cps=(64, 32, 16, 8, 4),
    downsample=(16, 8, 4, 2, 1),
    epoch_caps=(800, 600, 400, 400, 400),
    lrs=(0.03, 0.03, 0.01, 0.01, 0.003),
    early_stop_enabled=True,
    rule="stage_local_dual_median_relative_improvement_v1",
    check_interval_epochs=25,
    window_epochs=25,
    minimum_epochs=(120, 90, 60, 60, 60),
    configured_patience_epochs=(50, 38, 25, 25, 25),
    effective_patience_epochs=(50, 38, 25, 25, 25),
    total_relative_improvement_thresholds=(5.0e-5,) * 5,
    data_relative_improvement_thresholds=(2.0e-5,) * 5,
)


@dataclass(frozen=True)
class AdaptiveStopDecision:
    """One stage-local decision; no decision means continue."""

    stop: bool
    reason: str | None
    total_relative_improvement: float | None
    data_relative_improvement: float | None
    recent_total_median: float | None
    previous_total_median: float | None
    recent_data_median: float | None
    previous_data_median: float | None


def adaptive_stop_decision(
    rows: Sequence[Mapping[str, float]],
    schedule: ScheduleConfig,
    stage_index: int,
) -> AdaptiveStopDecision:
    """Implement the frozen OASIS dual-median stage-local stop rule.

    A check is only meaningful at the configured interval after the stage
    minimum and once enough samples exist for the preceding and recent
    windows.  Both total and data loss must show less improvement than their
    independent thresholds.  This intentionally never compares histories
    across variants.
    """

    if not schedule.early_stop_enabled:
        return AdaptiveStopDecision(False, None, None, None, None, None, None, None)
    epoch = len(rows)
    if epoch < schedule.minimum_epochs[stage_index] or epoch % schedule.check_interval_epochs != 0:
        return AdaptiveStopDecision(False, None, None, None, None, None, None, None)
    window = max(1, schedule.window_epochs)
    patience = max(window, schedule.effective_patience_epochs[stage_index])
    if len(rows) < patience + window:
        return AdaptiveStopDecision(False, None, None, None, None, None, None, None)
    recent_rows = rows[-window:]
    previous_stop = len(rows) - patience
    previous_start = max(0, previous_stop - window)
    previous_rows = rows[previous_start:previous_stop]
    if len(previous_rows) != window:
        return AdaptiveStopDecision(False, None, None, None, None, None, None, None)
    recent_total = float(np.median([float(row["loss"]) for row in recent_rows]))
    prior_total = float(np.median([float(row["loss"]) for row in previous_rows]))
    recent_data = float(np.median([float(row["data_loss"]) for row in recent_rows]))
    prior_data = float(np.median([float(row["data_loss"]) for row in previous_rows]))
    total_relative = (prior_total - recent_total) / max(abs(prior_total), 1.0e-8)
    data_relative = (prior_data - recent_data) / max(abs(prior_data), 1.0e-8)
    total_limit = schedule.total_relative_improvement_thresholds[stage_index]
    data_limit = schedule.data_relative_improvement_thresholds[stage_index]
    stop = total_relative < total_limit and data_relative < data_limit
    reason = None
    if stop:
        reason = (
            "adaptive_dual_median_relative_improvement_below_threshold"
            f";epoch={epoch};window={window};effective_patience={patience}"
            f";total_relative={total_relative:.9g};total_threshold={total_limit:.9g}"
            f";data_relative={data_relative:.9g};data_threshold={data_limit:.9g}"
        )
    return AdaptiveStopDecision(
        stop=stop,
        reason=reason,
        total_relative_improvement=float(total_relative),
        data_relative_improvement=float(data_relative),
        recent_total_median=recent_total,
        previous_total_median=prior_total,
        recent_data_median=recent_data,
        previous_data_median=prior_data,
    )
