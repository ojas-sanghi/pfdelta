#!/usr/bin/env python3
"""Visualize selected curves with seed aggregation and uncertainty bands.

Purpose:
This script is the "aggregation + rendering" stage of the pipeline. It reads
seed-level curve points from `selected_runs_curve_points.csv` (produced by
`analyze_best_epoch.py`) and converts each
`(task, model, selection_case)` group into:
- one mean trajectory over epochs
- one per-epoch standard-deviation trajectory
- one shaded `mean +/- std` uncertainty band

Why aggregation is done here:
`analyze_best_epoch.py` intentionally writes seed-level rows to preserve full
provenance. Aggregation is deferred to this script so visualization choices
(per-seed vs mean/std) can evolve without changing selection artifacts.

High-level workflow:
1. Load and type-cast curve-point CSV rows.
2. Group rows by `(task, model, selection_case)`.
3. For each group:
   - align epochs across selected seeds
   - compute per-epoch mean and population std (`pstdev`)
4. Render grouped curves as:
   - Plotly HTML (split pages or combined page), or
   - Matplotlib PNG
5. In Plotly mode, inject interactive blur controls into output HTML.

Important aggregation detail:
Epoch alignment uses intersection across seeds. An epoch is included only if it
exists for *all* selected seeds in that group. This avoids biased means from
mixing different seed counts at different epochs.

Quick commands:
- `uv run scripts/analyze_best_epoch/visualize_best_epoch.py --backend plotly --plotly-layout split`
- `uv run scripts/analyze_best_epoch/visualize_best_epoch.py --backend plotly --plotly-layout combined`
- `uv run scripts/analyze_best_epoch/visualize_best_epoch.py --backend matplotlib`
- `bash scripts/analyze_best_epoch/deploy_plotly_html_to_vercel.sh`
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import html
import os
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    """Parse command-line options for visualization outputs.

    Returns:
        Namespace containing:
        - `curve_csv`: seed-level input curve CSV
        - `backend`: `plotly`, `matplotlib`, or auto-selection
        - `plotly_layout`: page organization when using Plotly
        - output destinations for html/site/png variants
        - `title`: figure title
    """
    here = Path(__file__).resolve().parent
    outputs = here / "outputs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--curve-csv",
        type=Path,
        default=outputs / "selected_runs_curve_points.csv",
        help="CSV from analyze_best_epoch.py (default: outputs/selected_runs_curve_points.csv)",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "plotly", "matplotlib"),
        default="auto",
        help="Visualization backend (default: auto)",
    )
    parser.add_argument(
        "--plotly-layout",
        choices=("split", "combined"),
        default="split",
        help="Plotly output style: split site (index + per-model pages) or one combined page (default: split)",
    )
    parser.add_argument(
        "--output-site-dir",
        type=Path,
        default=outputs / "selected_runs_site",
        help="Output directory for split plotly site mode",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=outputs / "selected_runs_curves.html",
        help="Output HTML path for combined plotly mode",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=outputs / "selected_runs_curves.png",
        help="Output PNG path for matplotlib mode",
    )
    parser.add_argument(
        "--title",
        default="Selected Runs: Epoch vs PBL Mean",
        help="Figure title",
    )
    return parser.parse_args()


def load_curve_rows(csv_path: Path) -> List[dict]:
    """Load and normalize seed-level curve rows from CSV.

    Type coercion:
        `epoch`, `seed` -> int
        `metric_value`, `selected_error` -> float

    Deterministic ordering:
        Rows are sorted by model/task/case/seed/epoch/path so downstream
        grouping and plotting are stable across runs.
    """
    rows: List[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["epoch"] = int(row["epoch"])
            row["metric_value"] = float(row["metric_value"])
            row["selected_error"] = float(row["selected_error"])
            row["seed"] = int(row["seed"])
            rows.append(row)

    rows.sort(
        key=lambda r: (
            r["model"],
            r["task"],
            r["selection_case"],
            r["seed"],
            r["epoch"],
            r["run_path"],
        )
    )
    return rows


def _positive_floor(values: List[float]) -> float:
    """Return a small positive floor for log-scale-safe lower uncertainty.

    Since plots use log-y, lower bounds must stay > 0. We derive a conservative
    positive floor from observed means and clamp `mean - std` to that floor.
    """
    positives = [v for v in values if v > 0]
    if not positives:
        return 1e-12
    return max(min(positives) * 1e-6, 1e-12)


def aggregate_rows(rows: List[dict]) -> Dict[str, dict]:
    """Aggregate seed-level rows into mean/std curves per selection group.

    Grouping key:
        `(task, model, selection_case)`

    Aggregation math per group:
        Let `S` be selected seeds in the group.
        Let `E = intersection(epochs(seed) for seed in S)`.
        For each epoch `e in sorted(E)`:
            values_e = [loss(seed, e) for seed in S]
            mean_e = mean(values_e)
            std_e = pstdev(values_e)         # population std across seeds
            lower_e = max(mean_e - std_e, floor)
            upper_e = max(mean_e + std_e, floor)

    Why intersection (not union):
        It ensures every aggregated epoch uses the same seed count, so epoch-to-
        epoch changes in mean/std are not artifacts of missing seed logs.

    Returns:
        Mapping `group_id -> aggregated curve payload` where each payload
        contains arrays for epochs, mean/std, lower/upper bands, and metadata
        used for legends and hover text.
    """
    tmp: Dict[Tuple[str, str, str], dict] = {}

    for row in rows:
        key = (row["task"], row["model"], row["selection_case"])
        if key not in tmp:
            tmp[key] = {
                "task": row["task"],
                "model": row["model"],
                "selection_case": row["selection_case"],
                "seed_points": defaultdict(dict),  # seed -> {epoch: value}
                "selected_errors": {},
                "summary_best_points": {},
                "run_paths": {},
            }

        seed = int(row["seed"])
        tmp[key]["seed_points"][seed][row["epoch"]] = row["metric_value"]
        tmp[key]["selected_errors"][seed] = row["selected_error"]
        tmp[key]["summary_best_points"][seed] = row["summary_best_point"]
        tmp[key]["run_paths"][seed] = row["run_path"]

    grouped: Dict[str, dict] = {}
    for (task, model, selection_case), group in tmp.items():
        seeds = sorted(group["seed_points"])
        if not seeds:
            continue

        epoch_sets = [set(group["seed_points"][seed]) for seed in seeds]
        if not epoch_sets:
            continue

        # Keep only epochs observed in every selected seed for this group.
        common_epochs = sorted(set.intersection(*epoch_sets))
        if not common_epochs:
            # Keep deterministic fallback if no full overlap exists.
            union_epochs = sorted(set().union(*epoch_sets))
            common_epochs = [
                epoch
                for epoch in union_epochs
                if all(epoch in group["seed_points"][seed] for seed in seeds)
            ]
        if not common_epochs:
            continue

        mean_values: List[float] = []
        std_values: List[float] = []
        for epoch in common_epochs:
            # Epoch-wise aggregation across seeds.
            vals = [group["seed_points"][seed][epoch] for seed in seeds]
            mean_values.append(mean(vals))
            std_values.append(pstdev(vals) if len(vals) > 1 else 0.0)

        floor = _positive_floor(mean_values)
        lower_values = [max(mu - sigma, floor) for mu, sigma in zip(mean_values, std_values)]
        upper_values = [max(mu + sigma, floor) for mu, sigma in zip(mean_values, std_values)]

        selected_errors = [group["selected_errors"][seed] for seed in seeds]
        selected_error_mean = mean(selected_errors)
        selected_error_std = pstdev(selected_errors) if len(selected_errors) > 1 else 0.0
        seed_list = ",".join(str(seed) for seed in seeds)

        seed_run_summary = "; ".join(
            (
                f"seed={seed} "
                f"(bp={group['summary_best_points'][seed]}, "
                f"err={group['selected_errors'][seed]:.6f})"
            )
            for seed in seeds
        )

        display_label = (
            f"{task}|{selection_case}|mean(seed={seed_list})|"
            f"err_mean={selected_error_mean:.6f}"
        )
        group_id = f"{task}|{model}|{selection_case}"

        customdata = [
            [
                display_label,
                task,
                model,
                selection_case,
                seed_list,
                len(seeds),
                selected_error_mean,
                selected_error_std,
                std_values[idx],
                seed_run_summary,
            ]
            for idx, _epoch in enumerate(common_epochs)
        ]

        grouped[group_id] = {
            "task": task,
            "model": model,
            "selection_case": selection_case,
            "group_id": group_id,
            "display_label": display_label,
            "seeds": seeds,
            "epochs": common_epochs,
            "mean_values": mean_values,
            "std_values": std_values,
            "lower_values": lower_values,
            "upper_values": upper_values,
            "selected_error_mean": selected_error_mean,
            "selected_error_std": selected_error_std,
            "seed_run_summary": seed_run_summary,
            "customdata": customdata,
        }

    return grouped


def _model_ordered_curves(grouped: Dict[str, dict]) -> Dict[str, List[dict]]:
    """Return aggregated curves grouped by model with deterministic ordering."""
    by_model: Dict[str, List[dict]] = defaultdict(list)
    for curve in grouped.values():
        by_model[curve["model"]].append(curve)

    for model in sorted(by_model):
        by_model[model].sort(
            key=lambda curve: (
                curve["task"],
                curve["selection_case"],
                curve["display_label"],
            )
        )
    return by_model


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert '#RRGGBB' to 'rgba(r,g,b,a)'. Fallback to input color."""
    color = hex_color.strip()
    if color.startswith("#") and len(color) == 7:
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        return f"rgba({r}, {g}, {b}, {alpha:.3f})"
    return color


def _curve_traces(curve: dict, color: str):
    """Build `[upper, lower-fill, mean]` Plotly traces for one aggregated curve.

    Trace structure:
    - upper: invisible line used as the top boundary
    - lower: invisible line with `fill=tonexty` to create shaded std band
    - mean: visible center line with full hover metadata
    """
    import plotly.graph_objects as go

    hovertemplate = (
        "<b>%{customdata[0]}</b><br>"
        "epoch=%{x}<br>"
        "mean_loss=%{y:.6f}<br>"
        "std_loss=%{customdata[8]:.6f}<br>"
        "task=%{customdata[1]}<br>"
        "model=%{customdata[2]}<br>"
        "case=%{customdata[3]}<br>"
        "seeds=%{customdata[4]} (n=%{customdata[5]})<br>"
        "seed_error_mean=%{customdata[6]:.6f}<br>"
        "seed_error_std=%{customdata[7]:.6f}<br>"
        "seed_runs=%{customdata[9]}"
        "<extra></extra>"
    )

    # Use SVG Scatter for std bands so filled regions render reliably in HTML.
    upper = go.Scatter(
        x=curve["epochs"],
        y=curve["upper_values"],
        mode="lines",
        line={"width": 0},
        hoverinfo="skip",
        showlegend=False,
        legendgroup=curve["group_id"],
        meta={"trace_role": "std_upper"},
    )
    lower = go.Scatter(
        x=curve["epochs"],
        y=curve["lower_values"],
        mode="lines",
        line={"width": 0},
        fill="tonexty",
        fillcolor=_hex_to_rgba(color, 0.28),
        hoverinfo="skip",
        showlegend=False,
        legendgroup=curve["group_id"],
        meta={"trace_role": "std_lower"},
    )
    mean_line = go.Scatter(
        x=curve["epochs"],
        y=curve["mean_values"],
        mode="lines",
        name=curve["display_label"],
        legendgroup=curve["group_id"],
        line={"color": color, "width": 2.5},
        customdata=curve["customdata"],
        hovertemplate=hovertemplate,
        meta={"trace_role": "mean"},
    )
    return [upper, lower, mean_line]


def _blur_post_script(div_id: str) -> str:
    """Return JS for in-page Plotly interactions.

    Added interactions:
    - click curve: toggle blur state for that curve group
    - `B`: toggle hovered curve
    - reset button / `R`: restore all opacities
    """
    return f"""
(function() {{
  const gd = document.getElementById("{div_id}");
  if (!gd || gd.__curveBlurReady) return;
  gd.__curveBlurReady = true;

  const controlsId = "{div_id}_blur_controls";
  const resetId = "{div_id}_reset_blur";
  const baseOpacity = gd.data.map(() => 1.0);
  const blurredGroups = new Set();
  let hoveredCurve = null;

  function groupForCurve(curveNumber) {{
    const trace = gd.data[curveNumber];
    if (!trace) return null;
    return trace.legendgroup || ("trace-" + curveNumber);
  }}

  function ensureControls() {{
    if (document.getElementById(controlsId)) return;
    const controls = document.createElement("div");
    controls.id = controlsId;
    controls.style.cssText = "display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:8px 0 10px 0;font-size:13px;color:#334155;";
    controls.innerHTML =
      '<button id="' + resetId + '" type="button" style="border:1px solid #cbd5e1;background:#ffffff;color:#0f172a;padding:5px 10px;border-radius:6px;cursor:pointer;">Reset Blurred Curves</button>' +
      '<span>Click a curve to blur/unblur. Hover a curve + press <kbd>B</kbd>. Press <kbd>R</kbd> to reset.</span>' +
      '<span id="' + controlsId + '_status" style="color:#64748b;">state: ready</span>';
    if (gd.parentNode) {{
      gd.parentNode.insertBefore(controls, gd);
    }}
  }}

  function setStatus(text) {{
    const el = document.getElementById(controlsId + "_status");
    if (el) el.textContent = text;
  }}

  function setOpacities(opacityForIndex) {{
    for (let idx = 0; idx < gd.data.length; idx += 1) {{
      const trace = gd.data[idx];
      if (!trace) continue;
      trace.opacity = opacityForIndex(idx);
    }}
    Plotly.redraw(gd);
  }}

  function applyState() {{
    setOpacities((idx) => {{
      const groupId = groupForCurve(idx);
      if (groupId && blurredGroups.has(groupId)) {{
        return 0.12;
      }}
      return baseOpacity[idx];
    }});
    setStatus("state: blurred groups=" + blurredGroups.size);
  }}

  function toggleCurve(curveNumber) {{
    const groupId = groupForCurve(curveNumber);
    if (!groupId) return;
    if (blurredGroups.has(groupId)) {{
      blurredGroups.delete(groupId);
    }} else {{
      blurredGroups.add(groupId);
    }}
    applyState();
  }}

  function resetBlur() {{
    blurredGroups.clear();
    hoveredCurve = null;
    setOpacities(() => 1.0);
    setStatus("state: reset");
  }}

  function bindResetButton() {{
    const resetBtn = document.getElementById(resetId);
    if (!resetBtn) return;
    resetBtn.onclick = function(evt) {{
      if (evt) {{
        evt.preventDefault();
        evt.stopPropagation();
      }}
      resetBlur();
    }};
  }}

  ensureControls();
  bindResetButton();

  document.addEventListener("click", function(evt) {{
    if (!gd.isConnected) return;
    const target = evt && evt.target;
    if (target && target.id === resetId) {{
      evt.preventDefault();
      evt.stopPropagation();
      resetBlur();
    }}
  }});

  gd.on("plotly_click", function(evt) {{
    if (!evt || !evt.points || !evt.points.length) return;
    toggleCurve(evt.points[0].curveNumber);
    setStatus("state: toggled via click");
  }});

  gd.on("plotly_hover", function(evt) {{
    if (!evt || !evt.points || !evt.points.length) return;
    hoveredCurve = evt.points[0].curveNumber;
  }});

  window.addEventListener("keydown", function(evt) {{
    if (!gd.isConnected) return;
    const key = (evt.key || "").toLowerCase();
    if ((key === "b" || evt.code === "KeyB") && hoveredCurve !== null) {{
      toggleCurve(hoveredCurve);
      setStatus("state: toggled via keyboard B");
    }} else if (key === "r" || evt.code === "KeyR") {{
      resetBlur();
    }}
  }});
}})();
"""


def _make_model_plotly_figure(model: str, curves: List[dict], title: str):
    """Create one-model Plotly figure using aggregated mean/std traces."""
    import plotly.graph_objects as go
    from plotly.colors import qualitative

    case_order = sorted({curve["selection_case"] for curve in curves})
    palette = qualitative.Plotly + qualitative.Dark24 + qualitative.Light24
    color_by_case = {case: palette[i % len(palette)] for i, case in enumerate(case_order)}

    fig = go.Figure()
    for curve in curves:
        color = color_by_case[curve["selection_case"]]
        for trace in _curve_traces(curve, color):
            fig.add_trace(trace)

    fig.update_layout(
        title=f"{title} - {model}",
        template="plotly_white",
        height=860,
        width=1700,
        legend_title_text="Curve (task|case|mean-seeds|err_mean)",
        margin=dict(l=60, r=300, t=80, b=65),
    )
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Loss (PBL Mean, log scale)", type="log")

    return fig


def write_plotly_combined(grouped: Dict[str, dict], output_html: Path, title: str) -> None:
    """Render all models into one combined interactive Plotly HTML page."""
    from plotly.subplots import make_subplots
    from plotly.colors import qualitative

    by_model = _model_ordered_curves(grouped)
    model_order = sorted(by_model)
    case_order = sorted({curve["selection_case"] for curve in grouped.values()})
    palette = qualitative.Plotly + qualitative.Dark24 + qualitative.Light24
    color_by_case = {case: palette[i % len(palette)] for i, case in enumerate(case_order)}

    fig = make_subplots(
        rows=len(model_order),
        cols=1,
        shared_xaxes=False,
        subplot_titles=model_order,
        vertical_spacing=0.08,
    )

    for row_idx, model in enumerate(model_order, start=1):
        for curve in by_model[model]:
            color = color_by_case[curve["selection_case"]]
            for trace in _curve_traces(curve, color):
                fig.add_trace(trace, row=row_idx, col=1)

        fig.update_xaxes(title_text="Epoch", row=row_idx, col=1)
        fig.update_yaxes(title_text="Loss (PBL Mean, log scale)", type="log", row=row_idx, col=1)

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=max(700, 420 * len(model_order)),
        width=1800,
        legend_title_text="Curve (task|case|mean-seeds|err_mean)",
        margin=dict(l=50, r=40, t=90, b=60),
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        output_html,
        full_html=True,
        include_plotlyjs=True,
        div_id="best_epoch_curves",
        post_script=_blur_post_script("best_epoch_curves"),
    )


def write_plotly_split_site(grouped: Dict[str, dict], output_site_dir: Path, title: str) -> List[Path]:
    """Write split Plotly site: landing page + one page per model.

    Returns:
        Sorted list of written file paths.
    """
    by_model = _model_ordered_curves(grouped)
    model_order = sorted(by_model)

    output_site_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    for model in model_order:
        model_html = output_site_dir / f"{model}.html"
        div_id = f"curves_{model}"
        fig = _make_model_plotly_figure(model, by_model[model], title)
        fig_html = fig.to_html(
            full_html=False,
            include_plotlyjs=True,
            div_id=div_id,
            post_script=_blur_post_script(div_id),
        )

        nav_items: List[str] = ['<a href="index.html">Home</a>']
        for other_model in model_order:
            label = html.escape(other_model.title())
            if other_model == model:
                nav_items.append(f"<span class=\"current\">{label}</span>")
            else:
                nav_items.append(f"<a href=\"{html.escape(other_model)}.html\">{label}</a>")

        model_page_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)} - {html.escape(model)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f8fafc; }}
    .topbar {{ background: #ffffff; border-bottom: 1px solid #e2e8f0; padding: 10px 18px; position: sticky; top: 0; z-index: 10; }}
    .nav {{ display: flex; gap: 14px; align-items: center; flex-wrap: wrap; font-size: 14px; }}
    .nav a {{ color: #1d4ed8; text-decoration: none; }}
    .nav a:hover {{ text-decoration: underline; }}
    .nav .current {{ color: #111827; font-weight: 600; }}
    .meta {{ color: #475569; font-size: 13px; padding: 10px 18px 0 18px; }}
    .controls {{ padding: 8px 18px 0 18px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; font-size: 13px; color: #334155; }}
    .controls button {{ border: 1px solid #cbd5e1; background: #ffffff; color: #0f172a; padding: 5px 10px; border-radius: 6px; cursor: pointer; }}
    .controls kbd {{ background: #e2e8f0; border: 1px solid #cbd5e1; border-radius: 4px; padding: 1px 5px; font-size: 12px; }}
    .plot-wrap {{ padding: 0 10px 10px 10px; }}
  </style>
</head>
<body>
  <div class="topbar">
    <div class="nav">
      {' '.join(nav_items)}
    </div>
  </div>
  <div class="meta">
    Model: <strong>{html.escape(model)}</strong> | Curves: {len(by_model[model])} | Scale: log-y
  </div>
  <div class="controls" id="{div_id}_blur_controls">
    <button id="{div_id}_reset_blur" type="button">Reset Blurred Curves</button>
    <span>Click a curve to blur/unblur. Hover a curve + press <kbd>B</kbd>. Press <kbd>R</kbd> to reset.</span>
  </div>
  <div class="plot-wrap">
    {fig_html}
  </div>
</body>
</html>
"""
        model_html.write_text(model_page_html, encoding="utf-8")
        written.append(model_html)

    card_items: List[str] = []
    for model in model_order:
        n_curves = len(by_model[model])
        n_points = sum(len(curve["epochs"]) for curve in by_model[model])
        card_items.append(
            (
                f"<li><a href=\"{html.escape(model)}.html\">{html.escape(model.title())}</a>"
                f" - {n_curves} curves, {n_points} mean points</li>"
            )
        )

    index_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)} - Home</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; line-height: 1.45; }}
    h1 {{ margin-bottom: 0.4rem; }}
    .muted {{ color: #4b5563; }}
    .panel {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px 20px; max-width: 860px; }}
    a {{ color: #1d4ed8; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    code {{ background: #eef2ff; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p class="muted">Landing page for model-specific interactive plots.</p>
  <div class="panel">
    <h2>Model Pages</h2>
    <ul>
      {''.join(card_items)}
    </ul>
    <h3>How to Regenerate</h3>
    <p><code>uv run scripts/analyze_best_epoch/visualize_best_epoch.py --backend plotly --plotly-layout split</code></p>
  </div>
</body>
</html>
"""

    index_path = output_site_dir / "index.html"
    index_path.write_text(index_html, encoding="utf-8")
    written.append(index_path)

    return sorted(written, key=lambda p: p.as_posix())


def write_matplotlib(grouped: Dict[str, dict], output_png: Path, title: str) -> None:
    """Render grouped mean/std curves to static Matplotlib PNG.

    Visualization style mirrors Plotly semantics:
    - center line = mean
    - shaded band = mean +/- std
    """
    os.environ.setdefault("MPLCONFIGDIR", str((output_png.parent / ".mpl-cache").resolve()))
    import matplotlib.pyplot as plt

    by_model = _model_ordered_curves(grouped)
    model_order = sorted(by_model)

    fig, axes = plt.subplots(len(model_order), 1, figsize=(22, 8 * len(model_order)), squeeze=False)

    for idx, model in enumerate(model_order):
        ax = axes[idx][0]
        curves = by_model[model]
        case_order = sorted({curve["selection_case"] for curve in curves})
        cmap = plt.get_cmap("tab20")
        color_by_case = {
            case: cmap(case_idx % cmap.N)
            for case_idx, case in enumerate(case_order)
        }

        for curve in curves:
            color = color_by_case[curve["selection_case"]]
            ax.fill_between(
                curve["epochs"],
                curve["lower_values"],
                curve["upper_values"],
                color=color,
                alpha=0.16,
                linewidth=0,
            )
            ax.plot(
                curve["epochs"],
                curve["mean_values"],
                linewidth=2.0,
                color=color,
                label=curve["display_label"],
            )

        ax.set_title(model)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (PBL Mean, log scale)")
        ax.set_yscale("log", nonpositive="clip")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 0.78, 0.97])

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main() -> int:
    """Execute aggregation + visualization pipeline.

    Backend behavior:
    - `plotly`:
      - `split`: writes `index.html` + one model page each
      - `combined`: writes a single combined HTML
    - `matplotlib`: writes static PNG
    - `auto`: prefers plotly when installed
    """
    args = parse_args()
    rows = load_curve_rows(args.curve_csv.resolve())
    grouped = aggregate_rows(rows)

    if not grouped:
        raise RuntimeError(f"No rows found in {args.curve_csv}")

    backend = args.backend
    if backend == "auto":
        try:
            import plotly  # noqa: F401

            backend = "plotly"
        except ImportError:
            backend = "matplotlib"

    if backend == "plotly":
        if args.plotly_layout == "split":
            written = write_plotly_split_site(grouped, args.output_site_dir.resolve(), args.title)
            print("backend=plotly")
            print("plotly_layout=split")
            for path in written:
                print(f"wrote={path}")
        else:
            write_plotly_combined(grouped, args.output_html.resolve(), args.title)
            print("backend=plotly")
            print("plotly_layout=combined")
            print(f"wrote={args.output_html.resolve()}")
    else:
        write_matplotlib(grouped, args.output_png.resolve(), args.title)
        print("backend=matplotlib")
        print(f"wrote={args.output_png.resolve()}")

    print(f"curves={len(grouped)}")
    print(f"points={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
