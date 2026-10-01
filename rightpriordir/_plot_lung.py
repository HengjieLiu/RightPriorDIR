"""Audited registration/data kernels; see docs/methods.md for protocol scope."""

from __future__ import annotations
import math
from typing import Any, Mapping, Sequence
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

X_COL = "fg_logj_std_mean"

Y_COL = "original_snap_tre_mean_mm_mean"

X_STD_COL = "fg_logj_std_std"

Y_STD_COL = "original_snap_tre_mean_mm_std"

DAGGER = "†"

X_AXIS_LABEL = "Foreground SDlogJ"

ALL_METHODS = [
    "idir_be_default_lungmask",
    "dual_inr_pl1a_author_blur_native_loss",
    "sinr_cps4_global_ncc_fd_be",
    "direct_cp_cps4_global_ncc_fd_be",
    "direct_cp_32_16_8_4_global_ncc_fd_be",
    "direct_cp_32_16_8_4_ptv_lcc_isotv",
    "ptvreg_reference",
]

SELECTED_METHODS = [
    "idir_be_default_lungmask",
    "direct_cp_32_16_8_4_global_ncc_fd_be",
    "direct_cp_32_16_8_4_ptv_lcc_isotv",
    "ptvreg_reference",
]

FINAL_METHOD_SPECS = {
    "idir_be_default_lungmask": {
        "label": "INR-Dense",
        "color": "#0072B2",
        "marker": "o",
        "line": True,
    },
    "dual_inr_pl1a_author_blur_native_loss": {
        "label": "MR-INR-Dense",
        "color": "#56B4E9",
        "marker": "o",
        "line": True,
    },
    "sinr_cps4_global_ncc_fd_be": {
        "label": "INR-BSCP (cps4)",
        "color": "#009E73",
        "marker": "o",
        "line": True,
    },
    "direct_cp_cps4_global_ncc_fd_be": {
        "label": "D-BSCP (cps4)",
        "color": "#66C2A5",
        "marker": "o",
        "line": True,
    },
    "direct_cp_32_16_8_4_global_ncc_fd_be": {
        "label": "MR-D-BSCP (32-16-8-4)",
        "color": "#7E57C2",
        "marker": "o",
        "line": True,
    },
    "direct_cp_32_16_8_4_ptv_lcc_isotv": {
        "label": f"MR-D-BSCP (32-16-8-4){DAGGER}",
        "color": "#D55E00",
        "marker": "o",
        "line": True,
    },
    "ptvreg_reference": {
        "label": f"pTVReg{DAGGER}",
        "color": "#111111",
        "marker": "*",
        "line": False,
    },
}

AXIS_POLICY = {
    "xlim": (0.06, 0.5),
    "ylim": (0.6, 2.05),
    "xticks": [0.1, 0.2, 0.3, 0.4, 0.5],
    "yticks": [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
}

STYLE = {
    "figsize": (14.0, 10.0),
    "dpi": 300,
    "title_fontsize": 14,
    "axis_label_fontsize": 14,
    "tick_label_fontsize": 12,
    "legend_fontsize": 7.3,
    "legend_title_fontsize": 7.1,
    "marker_size": 48,
    "ptvreg_marker_size": 120,
    "line_width": 1.55,
    "ellipse_alpha": 0.12,
    "grid_alpha": 0.25,
    "wspace": 0.2,
    "hspace": 0.3,
}


def finite_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def finite_pair_rows(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out[X_COL] = pd.to_numeric(out[X_COL], errors="coerce")
    out[Y_COL] = pd.to_numeric(out[Y_COL], errors="coerce")
    out = out[out[X_COL].notna() & out[Y_COL].notna()].copy()
    if out.empty:
        return out
    out["sort_ratio"] = pd.to_numeric(
        out.get("ratio_to_best_tre_weight"), errors="coerce"
    )
    out["sort_reg"] = pd.to_numeric(out.get("reg_value"), errors="coerce")
    return out.sort_values(["sort_ratio", "sort_reg"], na_position="last")


def annotation_text(row: Mapping[str, Any]) -> str:
    method = str(row.get("method_id"))
    if method == "ptvreg_reference":
        return f"pTVReg{DAGGER}"
    label = str(row.get("reg_label") or "")
    return label if label else str(row.get("reg_value", ""))


def draw_lung_panel(
    ax: plt.Axes,
    rows: pd.DataFrame,
    methods: Sequence[str],
    *,
    title: str,
    annotate: bool,
    ellipses: bool,
) -> None:
    panel = rows[rows["method_id"].astype(str).isin(methods)].copy()
    for method in methods:
        group = finite_pair_rows(panel[panel["method_id"].astype(str).eq(method)])
        if group.empty:
            continue
        spec = FINAL_METHOD_SPECS[method]
        color = str(spec["color"])
        marker = str(spec["marker"])
        xs = group[X_COL].to_numpy(dtype=float)
        ys = group[Y_COL].to_numpy(dtype=float)
        if bool(spec["line"]) and len(group) > 1:
            ax.plot(
                xs, ys, color=color, linewidth=STYLE["line_width"], alpha=0.78, zorder=2
            )
        if ellipses:
            for _, row in group.iterrows():
                xerr = max(finite_float(row.get(X_STD_COL), 0.0), 0.0)
                yerr = max(finite_float(row.get(Y_STD_COL), 0.0), 0.0)
                if xerr > 0 or yerr > 0:
                    ax.add_patch(
                        Ellipse(
                            (finite_float(row[X_COL]), finite_float(row[Y_COL])),
                            width=2.0 * xerr,
                            height=2.0 * yerr,
                            facecolor=color,
                            edgecolor=color,
                            linewidth=0.9,
                            alpha=STYLE["ellipse_alpha"],
                            zorder=1,
                        )
                    )
        size = (
            STYLE["ptvreg_marker_size"]
            if method == "ptvreg_reference"
            else STYLE["marker_size"]
        )
        ax.scatter(
            xs,
            ys,
            s=size,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.7,
            label=str(spec["label"]),
            zorder=3,
        )
        if annotate:
            for _, row in group.iterrows():
                ax.annotate(
                    annotation_text(row),
                    (finite_float(row[X_COL]), finite_float(row[Y_COL])),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=6.2,
                    color=color,
                    clip_on=True,
                )
    ax.set_title(title, fontsize=STYLE["title_fontsize"])
    ax.set_xlabel(X_AXIS_LABEL)
    ax.set_ylabel("TRE (mm)")
    ax.set_xlim(*AXIS_POLICY["xlim"])
    ax.set_ylim(*AXIS_POLICY["ylim"])
    ax.set_xticks(AXIS_POLICY["xticks"])
    ax.set_yticks(AXIS_POLICY["yticks"])
    ax.tick_params(axis="both", labelsize=STYLE["tick_label_fontsize"])
    ax.xaxis.label.set_size(STYLE["axis_label_fontsize"])
    ax.yaxis.label.set_size(STYLE["axis_label_fontsize"])
    ax.grid(True, alpha=STYLE["grid_alpha"])
    (handles, labels) = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], linestyle="none", marker="", color="none"))
    labels.append(f"{DAGGER} pTVReg loss")
    legend = ax.legend(
        handles, labels, loc="best", fontsize=STYLE["legend_fontsize"], frameon=True
    )
    legend_texts = legend.get_texts()
    if legend_texts:
        legend_texts[-1].set_fontsize(STYLE["legend_title_fontsize"])
