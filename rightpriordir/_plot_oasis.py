"""Audited registration/data kernels; see docs/methods.md for protocol scope."""

from __future__ import annotations
import math
from typing import Dict, List, Mapping, Sequence, Tuple
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

IDIR_ALPHAS = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]

SPLINE_ALPHAS = [0.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]

SWEEP_METHODS = [
    "idir_original_dense_siren_be_autograd",
    "sinr_cps4_be_bspline_analytic",
    "direct_cp_cps4_be_bspline_analytic",
    "sinr_cps2_be_bspline_analytic",
    "direct_cp_cps2_be_bspline_analytic",
]

BASELINE_METHODS = [
    "bs1_vxm_v0_2",
    "bs2_tm_v0",
    "bs3_vfa_v0",
    "bs4_sitreg_v0",
    "greedy",
]

METHOD_SPECS: Dict[str, Dict[str, object]] = {
    "idir_original_dense_siren_be_autograd": {
        "label": "IDIR",
        "dot_label": "IDIR",
        "lr": 0.0001,
        "alphas": IDIR_ALPHAS,
        "color": "#005AB5",
    },
    "sinr_cps4_be_bspline_analytic": {
        "label": "SINR cps4",
        "dot_label": "SINR4",
        "lr": 0.0001,
        "alphas": SPLINE_ALPHAS,
        "color": "#009E73",
    },
    "direct_cp_cps4_be_bspline_analytic": {
        "label": "DirectCP cps4",
        "dot_label": "DCP4",
        "lr": 0.001,
        "alphas": SPLINE_ALPHAS,
        "color": "#66CDAA",
    },
    "sinr_cps2_be_bspline_analytic": {
        "label": "SINR cps2",
        "dot_label": "SINR2",
        "lr": 0.0001,
        "alphas": SPLINE_ALPHAS,
        "color": "#AA3377",
    },
    "direct_cp_cps2_be_bspline_analytic": {
        "label": "DirectCP cps2",
        "dot_label": "DCP2",
        "lr": 0.001,
        "alphas": SPLINE_ALPHAS,
        "color": "#EE77BB",
    },
    "bs1_vxm_v0_2": {"label": "BS1 VoxelMorph", "dot_label": "BS1", "color": "#F4A3A3"},
    "bs2_tm_v0": {"label": "BS2 TransMorph", "dot_label": "BS2", "color": "#E06666"},
    "bs3_vfa_v0": {"label": "BS3 VFA", "dot_label": "BS3", "color": "#B22222"},
    "bs4_sitreg_v0": {"label": "BS4 SITReg", "dot_label": "BS4", "color": "#8B0000"},
    "greedy": {"label": "Greedy", "dot_label": "Greedy", "color": "#D55E00"},
}

ACCURACY_METRICS: List[Tuple[str, str, bool]] = [
    ("dice_mean", "Mean Dice", True),
    ("hd95_mean", "Mean HD95", False),
    ("asd_warped_to_fixed_mean", "ASD warped-to-fixed", False),
    ("asd_symmetric_mean", "ASD symmetric", False),
]

REGULARITY_METRICS: List[Tuple[str, str]] = [
    ("be_fd_forward_fixed_fg_voxel", "FG BE forward FD"),
    ("diffusion_fd_forward_fixed_fg_voxel", "FG diffusion forward FD"),
    ("diffusion_l2_fg", "FG diffusion L2"),
    ("jdet_le0_pct_fg", "FG Jdet <= 0 (%)"),
    ("log_jdet_std_fg", "FG std(log Jdet)"),
    ("ndv_frac_fg", "FG NDV fraction"),
]


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def is_finite(value: object) -> bool:
    return math.isfinite(as_float(value))


def metric_mean(metric: str) -> str:
    return f"{metric}_mean"


def metric_std(metric: str) -> str:
    return f"{metric}_std"


def label_for_metric(metric: str, metrics: Sequence[Tuple[str, str]]) -> str:
    return dict(metrics)[metric]


def accuracy_label(metric: str) -> str:
    return {key: label for (key, label, _higher) in ACCURACY_METRICS}[metric]


def should_symlog(reg_key: str) -> bool:
    return (
        "jdet" not in reg_key and "ndv" not in reg_key and ("log_jdet" not in reg_key)
    )


def format_alpha(value: object) -> str:
    value_float = as_float(value)
    if not math.isfinite(value_float):
        return ""
    return f"{value_float:g}"


def add_ellipse(
    ax: plt.Axes,
    row: Mapping[str, object],
    x: float,
    y: float,
    reg_key: str,
    acc_key: str,
    color: str,
) -> None:
    x_std = max(as_float(row.get(metric_std(reg_key))), 0.0)
    y_std = max(as_float(row.get(metric_std(acc_key))), 0.0)
    if (
        not math.isfinite(x_std)
        or not math.isfinite(y_std)
        or (x_std == 0.0 and y_std == 0.0)
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
            alpha=0.14,
            zorder=1,
        )
    )


def collect_extents(
    rows: Sequence[Mapping[str, object]],
    reg_key: str,
    acc_key: str,
    *,
    include_std: bool,
) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for row in rows:
        x = as_float(row.get(metric_mean(reg_key)))
        y = as_float(row.get(metric_mean(acc_key)))
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        xs.append(x)
        ys.append(y)
        if include_std:
            x_std = max(as_float(row.get(metric_std(reg_key))), 0.0)
            y_std = max(as_float(row.get(metric_std(acc_key))), 0.0)
            if math.isfinite(x_std):
                xs.extend([x - x_std, x + x_std])
            if math.isfinite(y_std):
                ys.extend([y - y_std, y + y_std])
    return (xs, ys)


def set_limits(
    ax: plt.Axes,
    rows: Sequence[Mapping[str, object]],
    reg_key: str,
    acc_key: str,
    *,
    include_std: bool,
) -> None:
    (xs, ys) = collect_extents(rows, reg_key, acc_key, include_std=include_std)
    if not xs or not ys:
        return
    (x_min, x_max) = (min(xs), max(xs))
    (y_min, y_max) = (min(ys), max(ys))
    x_pad = 0.06 * max(x_max - x_min, 1e-08)
    y_pad = 0.06 * max(y_max - y_min, 1e-08)
    ax.set_xlim(max(0.0, x_min - x_pad), x_max + x_pad)
    ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)


def draw_panel(
    ax: plt.Axes,
    rows: Sequence[Dict[str, object]],
    *,
    acc_key: str,
    reg_key: str,
    style: str,
    show_legend: bool,
    annotate: bool,
) -> None:
    acc_mean = metric_mean(acc_key)
    reg_mean = metric_mean(reg_key)
    include_std = style in {"line_segments_errorbars", "ellipses"}
    for method in SWEEP_METHODS:
        group = [
            row
            for row in rows
            if str(row.get("method_key")) == method
            and is_finite(row.get(acc_mean))
            and is_finite(row.get(reg_mean))
        ]
        group.sort(key=lambda row: as_float(row.get("alpha")))
        if not group:
            continue
        color = str(METHOD_SPECS[method]["color"])
        label = str(METHOD_SPECS[method]["label"])
        xs = [as_float(row.get(reg_mean)) for row in group]
        ys = [as_float(row.get(acc_mean)) for row in group]
        if style == "ellipses":
            for row, x, y in zip(group, xs, ys):
                add_ellipse(ax, row, x, y, reg_key, acc_key, color)
        if style == "line_segments_errorbars":
            ax.errorbar(
                xs,
                ys,
                xerr=[
                    max(as_float(row.get(metric_std(reg_key))), 0.0) for row in group
                ],
                yerr=[
                    max(as_float(row.get(metric_std(acc_key))), 0.0) for row in group
                ],
                marker="o",
                linestyle="-",
                linewidth=1.55,
                markersize=4.6,
                capsize=2.0,
                color=color,
                label=label,
                zorder=3,
            )
        else:
            ax.plot(
                xs,
                ys,
                marker="o",
                linestyle="-",
                linewidth=1.55,
                markersize=4.8,
                color=color,
                label=label,
                zorder=3,
            )
        if annotate:
            for row, x, y in zip(group, xs, ys):
                ax.annotate(
                    format_alpha(row.get("alpha")),
                    (x, y),
                    fontsize=5.6,
                    xytext=(2, 2),
                    textcoords="offset points",
                )
    for method in BASELINE_METHODS:
        group = [
            row
            for row in rows
            if str(row.get("method_key")) == method
            and is_finite(row.get(acc_mean))
            and is_finite(row.get(reg_mean))
        ]
        if not group:
            continue
        row = group[0]
        color = str(METHOD_SPECS[method]["color"])
        label = str(METHOD_SPECS[method]["label"])
        x = as_float(row.get(reg_mean))
        y = as_float(row.get(acc_mean))
        marker_zorder = 12
        label_zorder = 13
        if style == "ellipses":
            add_ellipse(ax, row, x, y, reg_key, acc_key, color)
        if style == "line_segments_errorbars":
            ax.errorbar(
                [x],
                [y],
                xerr=[max(as_float(row.get(metric_std(reg_key))), 0.0)],
                yerr=[max(as_float(row.get(metric_std(acc_key))), 0.0)],
                marker="D",
                linestyle="none",
                markersize=5.2,
                capsize=2.0,
                color=color,
                markeredgecolor="#202020",
                markeredgewidth=0.45,
                label=label,
                zorder=marker_zorder,
            )
        else:
            ax.scatter(
                [x],
                [y],
                marker="D",
                s=42,
                color=color,
                edgecolors="#202020",
                linewidths=0.45,
                label=label,
                zorder=marker_zorder,
            )
        if annotate:
            ax.annotate(
                str(METHOD_SPECS[method]["dot_label"]),
                (x, y),
                fontsize=5.8,
                xytext=(4, 3),
                textcoords="offset points",
                zorder=label_zorder,
            )
    set_limits(ax, rows, reg_key, acc_key, include_std=include_std)
    if should_symlog(reg_key):
        ax.set_xscale("symlog", linthresh=1e-08)
    ax.set_xlabel(label_for_metric(reg_key, REGULARITY_METRICS))
    ax.set_ylabel(accuracy_label(acc_key))
    ax.grid(True, alpha=0.25)
    if show_legend:
        ax.legend(fontsize=6.4, ncol=2)
