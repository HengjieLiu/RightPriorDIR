"""Audited registration/data kernels; see docs/methods.md for protocol scope."""

from __future__ import annotations
import math
from typing import Any, Mapping
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

COMPOSITE_STYLE = {
    "figsize": (14.0, 10.0),
    "dpi": 300,
    "title_fontsize": 18,
    "axis_label_fontsize": 14,
    "tick_label_fontsize": 12,
    "legend_fontsize": 8.5,
    "legend_frame": True,
    "line_width": 1.55,
    "marker_size": 4.8,
    "lung_marker_size": 50,
    "ptvreg_marker_size": 120,
    "ellipse_alpha": 0.14,
    "lung_ellipse_alpha": 0.12,
    "grid_alpha": 0.25,
    "wspace": 0.18,
    "hspace": 0.3,
}


def _patch_oasis_composite_ellipse(oasis_plot: Any, alpha: float) -> None:

    def add_ellipse(
        ax: plt.Axes,
        row: Mapping[str, Any],
        x: float,
        y: float,
        reg_key: str,
        acc_key: str,
        color: str,
    ) -> None:
        x_std = max(oasis_plot.as_float(row.get(oasis_plot.metric_std(reg_key))), 0.0)
        y_std = max(oasis_plot.as_float(row.get(oasis_plot.metric_std(acc_key))), 0.0)
        if (
            not math.isfinite(x_std)
            or not math.isfinite(y_std)
            or (x_std <= 0 and y_std <= 0)
        ):
            return
        ax.add_patch(
            Ellipse(
                (x, y),
                width=2.0 * x_std,
                height=2.0 * y_std,
                facecolor=color,
                edgecolor=color,
                linewidth=0.9,
                alpha=alpha,
                zorder=1,
            )
        )

    oasis_plot.add_ellipse = add_ellipse


def _apply_composite_axis_style(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", labelsize=COMPOSITE_STYLE["tick_label_fontsize"])
    ax.xaxis.label.set_size(COMPOSITE_STYLE["axis_label_fontsize"])
    ax.yaxis.label.set_size(COMPOSITE_STYLE["axis_label_fontsize"])
    ax.grid(True, alpha=COMPOSITE_STYLE["grid_alpha"])


def _normalize_composite_legend(ax: plt.Axes, *, ncol: int) -> None:
    (handles, labels) = ax.get_legend_handles_labels()
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    seen = set()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)
    ax.legend(
        unique_handles,
        unique_labels,
        loc="best",
        ncol=ncol,
        fontsize=COMPOSITE_STYLE["legend_fontsize"],
        frameon=COMPOSITE_STYLE["legend_frame"],
    )


def _draw_composite_oasis_panel(
    ax: plt.Axes,
    oasis_plot: Any,
    oasis: pd.DataFrame,
    *,
    title: str,
    acc_key: str,
    annotate: bool,
) -> None:
    oasis_plot.REGULARITY_METRICS = [("log_jdet_std_fg", "SDlogJ")]
    _patch_oasis_composite_ellipse(oasis_plot, float(COMPOSITE_STYLE["ellipse_alpha"]))
    oasis_plot.draw_panel(
        ax,
        oasis.to_dict("records"),
        acc_key=acc_key,
        reg_key="log_jdet_std_fg",
        style="ellipses",
        show_legend=True,
        annotate=annotate,
    )
    ax.set_title(title, fontsize=COMPOSITE_STYLE["title_fontsize"])
    _apply_composite_axis_style(ax)
    _normalize_composite_legend(ax, ncol=2)
