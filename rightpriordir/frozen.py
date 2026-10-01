"""Recreate paper plots and release tables from bundled frozen CSVs on CPU."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def verify_inputs(root=ROOT):
    manifest = json.loads((root / "results/manifest.json").read_text())
    for item in manifest["inputs"]:
        path = root / item["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f'Frozen input checksum changed: {item["path"]}')
        if path.suffix == ".csv" and len(read_rows(path)) != item["row_count"]:
            raise ValueError(f'Frozen row count changed: {item["path"]}')
    return manifest


def md_table(headers, rows):
    return (
        "\n".join(
            [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
                *["| " + " | ".join(map(str, r)) + " |" for r in rows],
            ]
        )
        + "\n"
    )


def write_tables(out, root=ROOT):
    timing = read_rows(root / "results/updated_oasis/timing_first10.csv")
    groups = {}
    for r in timing:
        groups.setdefault(r["method"], []).append(r)
    table = []
    for rows in groups.values():
        if len(rows) != 2:
            raise ValueError("Expected paired base/fast rows")
        base = next(r for r in rows if "baseline" in r["candidate"])
        fast = next(r for r in rows if r is not base)
        f = lambda r, k: float(r[k])
        table.append(
            [
                base["method_label"],
                f"{f(base,'cold_endpoint_sec'):.2f} → {f(fast,'cold_endpoint_sec'):.2f}",
                f"{f(base,'warm_endpoint_mean_sec'):.2f} ± {f(base,'warm_endpoint_sd_sec'):.2f} → {f(fast,'warm_endpoint_mean_sec'):.2f} ± {f(fast,'warm_endpoint_sd_sec'):.2f}",
                f"{f(base,'warm_actual_epochs_mean'):.1f} → {f(fast,'warm_actual_epochs_mean'):.1f}",
                f"{f(base,'warm_endpoint_mean_sec')/f(fast,'warm_endpoint_mean_sec'):.2f}×",
            ]
        )
    (out / "updated_oasis_timing.md").write_text(
        "# Updated OASIS timing — September 16, 2026\n\nCold N=1; warm N=10. Loaded device images → saved DVF; not full-process runtime.\n\n"
        + md_table(
            [
                "Method",
                "Cold seconds (base → fast)",
                "Warm seconds, mean ± SD (base → fast)",
                "Warm steps (base → fast)",
                "Warm speed-up",
            ],
            table,
        )
    )
    accuracy = read_rows(root / "results/paper/oasis100_accuracy.csv")
    table = [
        [
            r["method_label"],
            str(r["cases"]),
            f"{float(r['baseline_dice']):.5f} → {float(r['fast_dice']):.5f}",
            f"{float(r['baseline_hd95']):.4f} → {float(r['fast_hd95']):.4f}",
        ]
        for r in accuracy
    ]
    (out / "historical_oasis100_accuracy.md").write_text(
        "# Historical OASIS100 accuracy (separate experiment)\n\nNot measured in the September 16 timing experiment.\n\n"
        + md_table(["Method", "N", "Dice (base → fast)", "HD95 (base → fast)"], table)
    )
    for source, name, title in [
        (
            "results/paper/table1_published.csv",
            "table1_published.md",
            "Published Table 1 — historical rounded values",
        )
    ]:
        rows = read_rows(root / source)
        keys = list(rows[0])
        (out / name).write_text(
            "# "
            + title
            + "\n\n"
            + md_table(keys, [[r[k] if r[k] else "—" for k in keys] for r in rows])
        )
    rows = read_rows(root / "results/copd_extension/population.csv")
    table = []
    for r in rows:
        metric = lambda key: (
            f"{float(r[key+'_mean']):.3f} ± {float(r[key+'_sample_sd']):.3f}"
            if r[key + "_mean"]
            else "—"
        )
        table.append(
            [
                r["method"],
                r["n_cases"],
                metric("headline_native_snap_mm"),
                metric("native_subvoxel_mm"),
                r["preprocessing"],
            ]
        )
    (out / "copd_extension.md").write_text(
        "# COPD — post-paper extension\n\nMean ± sample SD (ddof=1), mm. Protocol-tagged comparison; pTVReg uses the corrected MATLAB crop, other rows use legacy crop. Missing metrics remain missing.\n\n"
        + md_table(
            ["Method", "N", "Native snapped TRE", "Native subvoxel TRE", "Preparation"],
            table,
        )
    )


def figure2(out, root=ROOT):
    # Deliberately no torch or registration imports in the frozen-results path.
    import pandas as pd
    from . import _plot_oasis as op
    from . import _plot_style as style
    from . import _plot_lung as lp

    oasis = pd.read_csv(root / "results/paper/oasis100_arc.csv")
    lung = pd.read_csv(root / "results/paper/dirlab4dct_arc.csv")
    method = "direct_cp_8_4_2_be_bspline_analytic_multistage"
    if method not in op.SWEEP_METHODS:
        op.SWEEP_METHODS.append(method)
    op.METHOD_SPECS[method] = {
        "label": "MR-D-BSCP (8-4-2)",
        "dot_label": "MR-D-BSCP (8-4-2)",
        "color": "#332288",
    }
    for row in oasis.to_dict("records"):
        op.METHOD_SPECS[row["method_key"]]["label"] = row["display_method"]
    fig, axes = lp.plt.subplots(2, 2, figsize=lp.STYLE["figsize"], squeeze=False)
    for ax, key, title in [
        (axes[0, 0], "dice_mean", "Dice vs Foreground SDlogJ (OASIS N=100)"),
        (axes[0, 1], "hd95_mean", "HD95 vs Foreground SDlogJ (OASIS N=100)"),
    ]:
        style._draw_composite_oasis_panel(
            ax, op, oasis, title=title, acc_key=key, annotate=False
        )
        ax.set_xlabel("Foreground SDlogJ")
    lp.draw_lung_panel(
        axes[1, 0],
        lung,
        lp.ALL_METHODS,
        title="TRE vs Foreground SDlogJ (DIR-LAB 4DCT N=10), all",
        annotate=False,
        ellipses=False,
    )
    lp.draw_lung_panel(
        axes[1, 1],
        lung,
        lp.SELECTED_METHODS,
        title="TRE vs Foreground SDlogJ (DIR-LAB 4DCT N=10), selected",
        annotate=False,
        ellipses=True,
    )
    fig.subplots_adjust(wspace=lp.STYLE["wspace"], hspace=lp.STYLE["hspace"])
    for extension in ["png", "pdf"]:
        fig.savefig(
            out / f"figure2.{extension}", dpi=lp.STYLE["dpi"], bbox_inches="tight"
        )
    lp.plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tables-only", action="store_true")
    a = p.parse_args(argv)
    manifest = verify_inputs()
    a.out.mkdir(parents=True, exist_ok=True)
    targets = [
        "updated_oasis_timing.md",
        "historical_oasis100_accuracy.md",
        "table1_published.md",
        "copd_extension.md",
        "render_manifest.json",
    ]
    if not a.tables_only:
        targets += ["figure2.png", "figure2.pdf"]
    if any((a.out / n).exists() for n in targets):
        raise FileExistsError("Output exists; use a fresh --out directory")
    write_tables(a.out)
    if not a.tables_only:
        figure2(a.out)
    report = {
        "optimization_run": False,
        "raw_data_required": False,
        "inputs_verified": len(manifest["inputs"]),
        "files": {
            name: hashlib.sha256((a.out / name).read_bytes()).hexdigest()
            for name in targets
            if name != "render_manifest.json"
        },
    }
    (a.out / "render_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
