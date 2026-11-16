#!/usr/bin/env python3
"""Reproduce Section 6.2 with a ranking-flip strip and eight comparison panels.

Plot data are read from the packaged, verified original-RQ2 CSVs.
FbTw/DBLP/GrQc: five runs, mean and sample SD; lower panels include run dots.
Arxiv: one run; SD is unavailable and no error bars are drawn.

Examples:
    python plot_figure09.py
    python plot_figure09.py --no-tex
    python plot_figure09.py --width-cm 18.4 --height-cm 13.5
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bipga_plot"))
if shutil.which("latex") is None and Path("/Library/TeX/texbin").is_dir():
    os.environ["PATH"] = "/Library/TeX/texbin" + os.pathsep + os.environ.get("PATH", "")

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np

HERE = Path(__file__).resolve().parent
DATASETS = ["fbtw", "dblp", "grqc", "Arxiv1-Arxiv2"]
DATASET_NAMES = {"fbtw": "FbTw", "dblp": "DBLP", "grqc": "GrQc", "Arxiv1-Arxiv2": "Arxiv"}
METHODS = ["alpine", "fugal", "slotalign"]
METHOD_NAMES = {"alpine": "ALPINE", "fugal": "FUGAL", "slotalign": "SLOTAlign"}
REGIMES = ["F", "Us", "Ut", "B"]
METRICS = ["core_hits1", "decoded_core_recall"]
FIGURE_STEM = "figure_09"
STYLES = {
    "alpine": {"color": "#D55E00", "marker": "o"},
    "fugal": {"color": "#0072B2", "marker": "s"},
    "slotalign": {"color": "#56B4E9", "marker": ">"},
}

# Units: percent. None denotes unavailable SD, never an assumed zero SD.
# Figures use all common true correspondences, not only the lost pairs.
# Construction and algorithm seeds vary together in the five-run datasets.
def load_paper_observations(data_root=None):
    import csv
    root=Path(data_root) if data_root is not None else Path(__file__).resolve().parents[1]/'data'
    with (root/'figure_09/values.csv').open() as f:summary=list(csv.DictReader(f))
    with (root/'figure_09/values_by_run.csv').open() as f:runs=list(csv.DictReader(f))
    records=[]
    for row in summary:
        group=sorted((r for r in runs if all(r[k]==row[k] for k in ['dataset','method','setting'])),key=lambda r:int(r['seed']))
        records.append(dict(dataset=row['dataset'],method=row['method'],regime=row['setting'],metric={'hits1':'core_hits1','pair_recall':'decoded_core_recall'}[row['metric']],n_runs=int(row['n']),mean_pct=100*float(row['mean']),sd_pp=100*float(row['sample_sd']) if row['sample_sd'] else None,seeds=[int(r['seed']) for r in group],values_pct=[100*float(r[row['metric']]) for r in group]))
    with (root/'table_v/values.csv').open() as f:losses=list(csv.DictReader(f))
    flips=[dict(dataset=r['dataset'],method=r['method'],n_runs=int(r['n']),mean_pct=float(r['mean']),sd_pp=float(r['sample_sd']) if r['sample_sd'] else None) for r in losses if r['metric']=='top1_reversal_percent']
    return records,flips

PLOT_DATA,FLIP_DATA=[],[]




# Per-run r_flip values are averaged before computing sample SD; no pooling.
# Cohort: true pairs recovered in F, Us and Ut but not in B.
# Denominator per run: two directional queries for every lost pair.
# A reversal requires a unique true top-1 in F/Us/Ut, and a strictly
# higher-scoring incorrect candidate in B, within the fixed true candidates.
# These are the exact former Table V r_flip statistics.


def validated_lookup():
    lookup = {}
    expected = {(d, m, g, metric) for d in DATASETS for m in METHODS
                for g in REGIMES for metric in METRICS}
    for record in PLOT_DATA:
        key = tuple(record[k] for k in ("dataset", "method", "regime", "metric"))
        if key not in expected or key in lookup:
            raise ValueError(f"Duplicate or unexpected record: {key}")
        n = 1 if record["dataset"] == "Arxiv1-Arxiv2" else 5
        values = np.asarray(record["values_pct"], dtype=float)
        if record["n_runs"] != n or len(record["seeds"]) != n or values.shape != (n,):
            raise ValueError(f"Incorrect run count: {key}")
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 100)):
            raise ValueError(f"Invalid percentage values: {key}")
        if not np.isclose(values.mean(), record["mean_pct"], atol=1e-9, rtol=0):
            raise ValueError(f"Mean does not match individual runs: {key}")
        if n == 1:
            if record["sd_pp"] is not None:
                raise ValueError(f"A single run must not have an SD: {key}")
        elif not np.isclose(values.std(ddof=1), record["sd_pp"], atol=1e-9, rtol=0):
            raise ValueError(f"Sample SD does not match individual runs: {key}")
        lookup[key] = record
    if set(lookup) != expected:
        raise ValueError("Missing plotting records.")
    return lookup


def configure_style(use_tex):
    mpl.rcParams.update({
        "text.usetex": use_tex,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{amsmath}\usepackage{newtxtext,newtxmath}",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.linewidth": .65,
        "axes.unicode_minus": False,
        "xtick.labelsize": 8.6,
        "ytick.labelsize": 8.3,
        "xtick.major.width": .55,
        "ytick.major.width": .55,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "path",
        "savefig.facecolor": "white",
    })


def validated_flip_lookup():
    expected = {(d, m) for d in DATASETS for m in METHODS}
    lookup = {}
    for record in FLIP_DATA:
        key = record["dataset"], record["method"]
        if key not in expected or key in lookup:
            raise ValueError(f"Duplicate or unexpected flip record: {key}")
        n = 1 if record["dataset"] == "Arxiv1-Arxiv2" else 5
        mean, sd = record["mean_pct"], record["sd_pp"]
        if record["n_runs"] != n or not np.isfinite(mean) or not 0 <= mean <= 100:
            raise ValueError(f"Invalid flip record: {key}")
        if n == 1:
            if sd is not None:
                raise ValueError(f"Single-run Arxiv must not have an SD: {key}")
        elif sd is None or not np.isfinite(sd) or sd < 0:
            raise ValueError(f"Invalid sample SD: {key}")
        lookup[key] = record
    if set(lookup) != expected:
        raise ValueError("Missing flip records.")
    return lookup


def marker_options(method, size=3.9):
    style = STYLES[method]
    return dict(linestyle="none", color=style["color"],
                marker=style["marker"], markersize=size,
                markerfacecolor="white", markeredgewidth=1.0, zorder=4)


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D2D7DC", lw=.45, alpha=.8)
    ax.spines[["top", "right"]].set_visible(False)


def draw(lookup, flip_lookup, width_cm, height_cm, use_tex):
    """Top: lost-pair queries only. Lower two rows: all common true pairs."""
    fig = plt.figure(figsize=(width_cm / 2.54, height_cm / 2.54))
    left = .104 if not use_tex else .086
    right = .988
    pct = r"\%" if use_tex else "%"
    top_grid = fig.add_gridspec(1, 1, left=left, right=right,
                               bottom=.770, top=.890)
    core_grid = fig.add_gridspec(2, 4, left=left, right=right,
                                bottom=.170, top=.630,
                                hspace=.22, wspace=.14)

    # A separate axis keeps r_flip's cohort and x-axis distinct from the
    # four-condition comparisons below. It is not a fifth setting.
    flip_ax = fig.add_subplot(top_grid[0, 0])
    flip_ax.set_gid("lost_correspondences")
    style_axes(flip_ax)
    flip_ax.set_xlim(-.45, 3.45)
    # Do not clip sample-SD error bars that extend slightly above 100%.
    flip_hi = max(105, max(r["mean_pct"] + (r["sd_pp"] or 0)
                          for r in flip_lookup.values()) + 2)
    flip_ax.set_ylim(0, flip_hi)
    flip_ax.set_yticks([0, 50, 100])
    flip_ax.set_ylabel(r"$r_{\mathrm{flip}}$" + f" ({pct})", labelpad=6)
    flip_ax.set_title("(a) Lost correspondences: ranking reversals",
                      loc="left", fontsize=10, pad=7)
    flip_ax.set_xticks(range(4), [
        DATASET_NAMES[d] + (r" ($n=1$)" if d == "Arxiv1-Arxiv2" else "")
        for d in DATASETS
    ])
    flip_ax.tick_params(axis="x", labelsize=10, length=0, pad=5)
    for boundary in [.5, 1.5, 2.5]:
        flip_ax.axvline(boundary, color="#E2E6EA", lw=.5, zorder=0)
    flip_errorbars = flip_singles = 0
    for di, dataset in enumerate(DATASETS):
        for mi, method in enumerate(METHODS):
            record = flip_lookup[dataset, method]
            x = di + (mi - 1) * .18
            opts = marker_options(method, size=4.4)
            if record["n_runs"] == 1:
                artists = flip_ax.plot([x], [record["mean_pct"]], **opts)
                artists[0].set_gid(f"flip_single:{dataset}:{method}")
                flip_singles += 1
            else:
                container = flip_ax.errorbar(
                    [x], [record["mean_pct"]], yerr=[record["sd_pp"]],
                    elinewidth=.85, capsize=2.1, capthick=.8, **opts,
                )
                container[0].set_gid(f"flip_repeated:{dataset}:{method}")
                flip_errorbars += 1

    fig.text((left + right) / 2, .701,
             "All common true correspondences: fixed true candidate sets",
             ha="center", va="center", fontsize=9.2)
    axes = np.empty((2, 4), dtype=object)
    row_names = [f"Ranking ability\nHits@1 ({pct})",
                 f"Matching ability\nPair recall ({pct})"]
    lower, upper = [], []
    for record in lookup.values():
        sd = record["sd_pp"] or 0
        lower.append(record["mean_pct"] - sd)
        upper.append(record["mean_pct"] + sd)
    ylo = min(-3, float(np.floor(min(lower) - 2)))
    yhi = max(103, float(np.ceil(max(upper)) + 2))
    core_errorbars = core_singles = 0
    for ri, metric in enumerate(METRICS):
        for di, dataset in enumerate(DATASETS):
            shared = {} if (ri, di) == (0, 0) else dict(sharex=axes[0, 0], sharey=axes[0, 0])
            ax = fig.add_subplot(core_grid[ri, di], **shared)
            ax.set_gid(f"all_common:{dataset}:{metric}")
            axes[ri, di] = ax
            style_axes(ax)
            ax.axvspan(2.55, 3.46, color="#F1F3F5", zorder=0)
            ax.set_xlim(-.46, 3.46)
            ax.set_ylim(ylo, yhi)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_xticks(range(4), ["F", r"$U_s$", r"$U_t$", "B"])
            ax.tick_params(labelleft=(di == 0), labelbottom=(ri == 1))
            for mi, method in enumerate(METHODS):
                style = STYLES[method]
                xs = np.arange(4) + (mi - 1) * .235
                records = [lookup[dataset, method, regime, metric] for regime in REGIMES]
                means = [r["mean_pct"] for r in records]
                if dataset == "Arxiv1-Arxiv2":
                    artists = ax.plot(xs, means, **marker_options(method))
                    artists[0].set_gid(f"core_single:{dataset}:{method}:{metric}")
                    core_singles += len(records)
                else:
                    for gi, record in enumerate(records):
                        ax.scatter(
                            xs[gi] + np.linspace(-.055, .055, record["n_runs"]),
                            record["values_pct"], s=8, color=style["color"],
                            alpha=.50, linewidths=0, zorder=3,
                        )
                    ax.errorbar(xs, means, yerr=[r["sd_pp"] for r in records],
                                elinewidth=.8, capsize=1.8, capthick=.75,
                                **marker_options(method))
                    core_errorbars += len(records)
            if ri == 0:
                title = DATASET_NAMES[dataset]
                if dataset == "Arxiv1-Arxiv2":
                    title += r" ($n=1$)"
                ax.set_title(title, pad=6)
            if di == 0:
                ax.set_ylabel(row_names[ri], labelpad=6)
            ax.text(.025, .055, f"({chr(98 + ri * 4 + di)})",
                    transform=ax.transAxes, fontsize=8.5)

    handles = [Line2D([], [], label=METHOD_NAMES[m], linestyle="none",
                      marker=STYLES[m]["marker"], color=STYLES[m]["color"],
                      markersize=5, markerfacecolor="white", markeredgewidth=1.1)
               for m in METHODS]
    fig.legend(handles=handles, ncol=3, loc="upper center",
               bbox_to_anchor=((left + right) / 2, .995), frameon=False,
               columnspacing=2.2, handletextpad=.5, fontsize=10)
    settings = (r"F: FullGA $(1,1)$\qquad $U_s$: UniPGA $(0.8,1)$"
                r"\qquad $U_t$: UniPGA $(1,0.8)$\qquad B: BiPGA $(0.8,0.8)$")
    if not use_tex:
        settings = settings.replace(r"\qquad", "    ")
    fig.text(.5, .103, settings, ha="center", va="center", fontsize=8.2)
    fig.text(.5, .062,
             r"Small dots (lower panels): runs. Markers/bars: mean $\pm$ sample SD ($n=5$); Arxiv: one run.",
             ha="center", va="center", fontsize=7.5)
    fig.text(.5, .024,
             "Matching uses the same candidates, output size and one-to-one decoder.",
             ha="center", va="center", fontsize=8)
    if (flip_errorbars, flip_singles, core_errorbars, core_singles) != (9, 3, 72, 24):
        raise ValueError("Unexpected number of repeated/single-run plot records.")
    return fig


def check_canvas(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    overflow = []
    for artist in fig.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        box = artist.get_window_extent(renderer)
        if artist.axes is not None and not box.overlaps(bounds):
            continue
        if (box.x0 < bounds.x0 - 1 or box.y0 < bounds.y0 - 1
                or box.x1 > bounds.x1 + 1 or box.y1 > bounds.y1 + 1):
            overflow.append(artist.get_text())
    if overflow:
        raise ValueError(f"Text outside the figure canvas: {overflow}")


def main():
    global PLOT_DATA, FLIP_DATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "figures")
    parser.add_argument("--data-root", type=Path, default=HERE.parent / "data")
    parser.add_argument("--width-cm", type=float, default=18.4)
    parser.add_argument("--height-cm", type=float, default=13.5)
    parser.add_argument("--formats", nargs="+", choices=["pdf", "png", "svg"],
                        default=["pdf", "png"])
    parser.add_argument("--no-tex", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    PLOT_DATA, FLIP_DATA = load_paper_observations(args.data_root)
    lookup, flip_lookup = validated_lookup(), validated_flip_lookup()
    if args.check_only:
        print("Verified 96 core-comparison records and 12 ranking-flip records.")
        return
    if args.width_cm <= 0 or args.height_cm <= 0:
        parser.error("Figure dimensions must be positive.")
    if not args.no_tex and (shutil.which("latex") is None or shutil.which("dvipng") is None):
        parser.error("LaTeX/dvipng not found. Use --no-tex.")
    configure_style(not args.no_tex)
    fig = draw(lookup, flip_lookup, args.width_cm, args.height_cm, not args.no_tex)
    check_canvas(fig)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in args.formats:
        fig.savefig(args.output_dir / f"{FIGURE_STEM}.{extension}", dpi=300)
    plt.close(fig)
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()), "datasets": 4, "methods": 3,
        "panels": 9, "flip_records": 12, "core_comparison_records": 96,
        "arxiv_error_bars": False, "text_overflow": [],
    }, indent=2))


if __name__ == "__main__":
    main()
