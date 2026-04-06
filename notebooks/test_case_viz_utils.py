"""Utilities for the generic April test-case visualization notebook.

This module reads the real April run outputs directly from the run directories.
It does not depend on the batch-generation configs under ``core/configs``.
Instead, each run directory's ``config.yaml`` is treated as the source of truth
for dataset ordering, model identity, and metric naming.
"""

from __future__ import annotations

import html
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

try:
    import yaml
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "PyYAML is required for the April test-case visualization notebook."
    ) from exc


PBL_METRIC = "PBL Mean"
DEFAULT_EXPECTED_SEEDS = (42, 43, 44)
LOSS_FAMILY_SORT_PRIORITY = {"pbl": 0, "mse": 1}
SEED_RE = re.compile(r"seed=(?P<seed>\d+)")
TASK_RE = re.compile(r"^task(?P<num>\d+)$")
CASE_NUMBER_RE = re.compile(r"\d+")


def load_run_suite(
    run_root: str | Path,
    expected_seeds: tuple[int, ...] = DEFAULT_EXPECTED_SEEDS,
    skip_incomplete_combos: bool = False,
) -> dict[str, Any]:
    """Load the April run suite and organize it by ``(model, loss_family)``.

    When ``skip_incomplete_combos`` is ``True``, incomplete or inconsistent
    combos are recorded in ``skipped_items`` and omitted from the returned suite
    instead of aborting the entire load.
    """
    run_root_path = Path(run_root).resolve()
    if not run_root_path.is_dir():
        raise FileNotFoundError(f"Run root does not exist: {run_root_path}")

    normalized_expected_seeds = tuple(sorted(int(seed) for seed in expected_seeds))
    family_roots = _discover_family_roots(run_root_path)
    if not family_roots:
        raise ValueError(f"No loss-family directories were found under {run_root_path}")

    combos: dict[tuple[str, str], dict[str, Any]] = {}
    suite_test_cases: tuple[str, ...] | None = None
    skipped_items: list[dict[str, str]] = []

    for loss_family, family_root in family_roots:
        task_dirs = _sorted_task_dirs(family_root)
        if not task_dirs:
            raise ValueError(f"No task directories were found under {family_root}")

        for task_dir in task_dirs:
            model_dirs = sorted(path for path in task_dir.iterdir() if path.is_dir())
            if not model_dirs:
                raise ValueError(f"No model directories were found under {task_dir}")

            for model_dir in model_dirs:
                try:
                    selected_records = _select_best_records_for_model_task(
                        model_dir=model_dir,
                        task=task_dir.name,
                        loss_family=loss_family,
                        expected_seeds=normalized_expected_seeds,
                    )
                except Exception as exc:
                    if not skip_incomplete_combos:
                        raise
                    skipped_items.append(
                        {
                            "model_slug": _normalize_slug(model_dir.name),
                            "loss_family": loss_family,
                            "task": task_dir.name,
                            "reason": str(exc),
                        }
                    )
                    continue

                if not selected_records:
                    raise ValueError(f"No run records were selected under {model_dir}")

                combo_key = (selected_records[0]["model_slug"], loss_family)
                combo = combos.setdefault(
                    combo_key,
                    {
                        "model_slug": combo_key[0],
                        "loss_family": loss_family,
                        "records": [],
                        "folder_model_names": set(),
                    },
                )
                combo["records"].extend(selected_records)
                combo["folder_model_names"].add(model_dir.name)

                for record in selected_records:
                    record_test_cases = tuple(record["test_cases"])
                    if suite_test_cases is None:
                        suite_test_cases = record_test_cases
                    elif record_test_cases != suite_test_cases:
                        if not skip_incomplete_combos:
                            raise ValueError(
                                "Inconsistent grouped test-case order across the suite: "
                                f"expected {suite_test_cases}, found {record_test_cases} "
                                f"at {record['run_path']}"
                            )
                        skipped_items.append(
                            {
                                "model_slug": record["model_slug"],
                                "loss_family": loss_family,
                                "task": task_dir.name,
                                "reason": (
                                    "Inconsistent grouped test-case order across the suite: "
                                    f"expected {suite_test_cases}, found {record_test_cases} at {record['run_path']}"
                                ),
                            }
                        )
                        combo["records"] = []
                        break

    if suite_test_cases is None:
        detail = _format_skip_summary(skipped_items)
        raise ValueError(
            f"No validation test cases were discovered under {run_root_path}. {detail}".strip()
        )
    if not combos:
        detail = _format_skip_summary(skipped_items)
        raise ValueError(
            f"No model/loss-family combos were discovered under {run_root_path}. {detail}".strip()
        )

    prepared_combos: dict[tuple[str, str], dict[str, Any]] = {}
    for combo_key, combo in combos.items():
        if not combo["records"]:
            continue
        try:
            records = sorted(
                combo["records"],
                key=lambda record: (
                    _task_sort_key(record["task"]),
                    _case_sort_key(record["train_case"]),
                    record["seed"],
                ),
            )

            if len(combo["folder_model_names"]) > 1:
                raise ValueError(
                    f"Inconsistent model slug across tasks for {combo_key}: {sorted(combo['folder_model_names'])}"
                )

            selection_metric_key = _single_unique_value(
                {record["selection_metric_key"] for record in records},
                f"selection metric for combo {combo_key}",
            )
            pbl_metric_key = _single_unique_value(
                {record["pbl_metric_key"] for record in records},
                f"PBL metric for combo {combo_key}",
            )
            mse_metric_key = _single_unique_value(
                {record["mse_metric_key"] for record in records},
                f"MSE metric for combo {combo_key}",
            )
            signature = _group_signature(records)
            row_labels = sorted({record["train_case"] for record in records}, key=_case_sort_key)
            col_labels = list(suite_test_cases)
            display_name = f"{combo['model_slug']} - {combo['loss_family']}"
            audit_rows = [
                {
                    "model_slug": record["model_slug"],
                    "loss_family": record["loss_family"],
                    "task": record["task"],
                    "train_case": record["train_case"],
                    "seed": record["seed"],
                    "best_point": record["best_point"],
                    "selection_metric_key": record["selection_metric_key"],
                    "selection_metric_value": record["selection_metric_value"],
                    "run_path": record["run_path"],
                    "run_config_path": record["run_config_path"],
                }
                for record in records
            ]
        except Exception as exc:
            if not skip_incomplete_combos:
                raise
            skipped_items.append(
                {
                    "model_slug": combo_key[0],
                    "loss_family": combo_key[1],
                    "task": "*",
                    "reason": str(exc),
                }
            )
            continue

        prepared_combos[combo_key] = {
            **combo,
            "records": records,
            "display_name": display_name,
            "selection_metric_key": selection_metric_key,
            "pbl_metric_key": pbl_metric_key,
            "mse_metric_key": mse_metric_key,
            "row_labels": row_labels,
            "col_labels": col_labels,
            "expected_seeds": list(normalized_expected_seeds),
            "audit_rows": audit_rows,
            "signature": signature,
        }

    if not prepared_combos:
        detail = _format_skip_summary(skipped_items)
        raise ValueError(
            f"No complete model/loss-family combos were discovered under {run_root_path}. {detail}".strip()
        )

    ordered_prepared_keys = sorted(prepared_combos, key=lambda key: (-len(prepared_combos[key]["signature"]), _combo_sort_key(key)))
    reference_key = ordered_prepared_keys[0]
    reference_signature = prepared_combos[reference_key]["signature"]
    reference_combo_label = prepared_combos[reference_key]["display_name"]

    active_combos: dict[tuple[str, str], dict[str, Any]] = {}
    combo_summary_rows = []
    all_audit_rows = []

    for combo_key in sorted(prepared_combos, key=_combo_sort_key):
        combo = prepared_combos[combo_key]
        if combo["signature"] != reference_signature:
            message = (
                "Incomplete or inconsistent combo layout detected: "
                f"combo {combo['display_name']} has groups {combo['signature']}, "
                f"but {reference_combo_label} has {reference_signature}"
            )
            if not skip_incomplete_combos:
                raise ValueError(message)
            skipped_items.append(
                {
                    "model_slug": combo["model_slug"],
                    "loss_family": combo["loss_family"],
                    "task": "*",
                    "reason": message,
                }
            )
            continue

        combo.pop("signature", None)
        active_combos[combo_key] = combo
        combo_summary_rows.append(
            {
                "tab_label": combo["display_name"],
                "model_slug": combo["model_slug"],
                "loss_family": combo["loss_family"],
                "selection_metric_key": combo["selection_metric_key"],
                "pbl_metric_key": combo["pbl_metric_key"],
                "mse_metric_key": combo["mse_metric_key"],
                "num_records": len(combo["records"]),
                "num_groups": len(reference_signature),
            }
        )
        all_audit_rows.extend(combo["audit_rows"])

    if not active_combos:
        detail = _format_skip_summary(skipped_items)
        raise ValueError(
            f"No complete model/loss-family combos were discovered under {run_root_path}. {detail}".strip()
        )

    global_row_labels = sorted(
        {train_case for task, train_case in reference_signature},
        key=_case_sort_key,
    )

    return {
        "run_root": str(run_root_path),
        "expected_seeds": list(normalized_expected_seeds),
        "combo_order": sorted(active_combos, key=_combo_sort_key),
        "combos": active_combos,
        "row_labels": global_row_labels,
        "col_labels": list(suite_test_cases),
        "combo_summary_rows": combo_summary_rows,
        "audit_rows": all_audit_rows,
        "skipped_items": skipped_items,
    }


def load_run_family(
    run_root: str | Path,
    config_root: str | Path | None = None,
    primary_metric: str | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for older notebook imports."""
    del config_root, primary_metric
    suite_data = load_run_suite(run_root)
    if len(suite_data["combo_order"]) != 1:
        raise ValueError(
            "load_run_family only works when the supplied run root resolves to exactly "
            "one (model, loss_family) combo. Use load_run_suite instead."
        )
    combo_key = suite_data["combo_order"][0]
    return suite_data["combos"][combo_key]


def build_metric_matrices(
    combo_data: dict[str, Any] | list[dict[str, Any]],
    metric_key: str,
) -> dict[str, Any]:
    """Aggregate seed-level records into mean/std matrices for one metric."""
    if isinstance(combo_data, dict):
        records = combo_data["records"]
        row_labels = combo_data["row_labels"]
        col_labels = combo_data["col_labels"]
        expected_seeds = tuple(combo_data["expected_seeds"])
    else:
        records = combo_data
        row_labels = sorted({record["train_case"] for record in records}, key=_case_sort_key)
        first_record = records[0]
        col_labels = list(first_record["test_cases"])
        expected_seeds = DEFAULT_EXPECTED_SEEDS

    if not records:
        raise ValueError("No run records were provided.")

    grouped: dict[str, dict[str, list[float]]] = {
        train_case: {test_case: [] for test_case in col_labels}
        for train_case in row_labels
    }

    for record in records:
        train_case = record["train_case"]
        if train_case not in grouped:
            raise ValueError(f"Unexpected train case in records: {train_case}")

        for test_case in col_labels:
            if test_case not in record["test_case_metrics"]:
                raise ValueError(
                    f"Missing test case {test_case} in record {record['run_path']}"
                )
            case_metrics = record["test_case_metrics"][test_case]
            if metric_key not in case_metrics:
                raise KeyError(
                    f"Metric {metric_key} is missing from {record['run_path']} for {test_case}"
                )
            grouped[train_case][test_case].append(float(case_metrics[metric_key]))

    means = np.empty((len(row_labels), len(col_labels)), dtype=float)
    stds = np.empty((len(row_labels), len(col_labels)), dtype=float)
    seed_counts = np.empty((len(row_labels), len(col_labels)), dtype=int)

    for row_index, train_case in enumerate(row_labels):
        for col_index, test_case in enumerate(col_labels):
            values = grouped[train_case][test_case]
            if len(values) != len(expected_seeds):
                raise ValueError(
                    f"Expected exactly {len(expected_seeds)} values for {train_case} -> {test_case}, "
                    f"found {len(values)}"
                )
            means[row_index, col_index] = float(np.mean(values))
            stds[row_index, col_index] = float(np.std(values))
            seed_counts[row_index, col_index] = len(values)

    return {
        "metric_key": metric_key,
        "metric_label": metric_key,
        "means": means,
        "stds": stds,
        "row_labels": list(row_labels),
        "col_labels": list(col_labels),
        "seed_counts": seed_counts,
    }


def plot_grouped_bar_chart(
    means: np.ndarray,
    stds: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    metric_label: str,
    chart_title: str,
):
    """Render the notebook's static grouped-bar chart."""
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    _validate_matrix_shapes(means, stds, row_labels, col_labels)

    n_train, n_test = means.shape
    x = np.arange(n_test)
    bar_width = 0.8 / n_train
    fig, ax = plt.subplots(figsize=(15, 6))
    cmap = plt.colormaps["tab20"]

    for index, train_case in enumerate(row_labels):
        offset = (index - n_train / 2) * bar_width + bar_width / 2
        ax.bar(
            x + offset,
            means[index],
            width=bar_width,
            yerr=stds[index],
            label=_display_case(train_case),
            capsize=3,
            linewidth=0.8,
            edgecolor="black",
            alpha=0.95,
            color=cmap(index),
            error_kw={"elinewidth": 1, "capsize": 3, "capthick": 1},
        )

    ax.set_title(chart_title, pad=10, fontsize=18)
    ax.set_ylabel(f"Mean {metric_label} (+/- 1 SD) (log)", fontsize=14)
    ax.set_xlabel("Test Case", fontsize=14)
    ax.set_xticks(x, [_display_case(label) for label in col_labels], fontsize=12)
    ax.set_yscale("log")
    ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.6, alpha=0.35)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        title="Trained On",
        title_fontsize=12,
        fontsize=10,
    )

    try:
        fig.tight_layout()
    except ValueError:
        ax.set_yscale("linear")
        fig.tight_layout()

    plt.show()
    return fig


def plot_grouped_bar_chart_interactive(
    means: np.ndarray,
    stds: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    metric_label: str,
    chart_title: str,
):
    """Render the Plotly grouped-bar chart used for hoverable inspection."""
    go = _load_plotly_go()
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    _validate_matrix_shapes(means, stds, row_labels, col_labels)

    x_labels = [_display_case(label) for label in col_labels]
    cmap = plt.colormaps["tab20"]
    fig = go.Figure()

    for index, train_case in enumerate(row_labels):
        customdata = np.array(
            [
                [_display_case(train_case), float(stds[index, col]), metric_label]
                for col in range(len(col_labels))
            ],
            dtype=object,
        )
        fig.add_bar(
            x=x_labels,
            y=means[index],
            name=_display_case(train_case),
            marker_color=_to_rgba_string(cmap(index)),
            error_y={"type": "data", "array": stds[index], "visible": True, "thickness": 1.0, "width": 3},
            customdata=customdata,
            hovertemplate=(
                "Trained on: %{customdata[0]}<br>"
                "Metric: %{customdata[2]}<br>"
                "Mean: %{y:.6g}<br>"
                "Std Dev: %{customdata[1]:.6g}<extra></extra>"
            ),
        )

    fig.update_layout(
        title=chart_title,
        xaxis_title="Test Case",
        yaxis_title=f"Mean {metric_label} (+/- 1 SD) (log)",
        barmode="group",
        legend_title="Trained On",
        template="plotly_white",
        width=1500,
        height=650,
    )
    fig.update_yaxes(type="log", showgrid=True, gridcolor="rgba(0, 0, 0, 0.15)")
    fig.show()
    return fig


def plot_metric_toggle_chart(
    pbl_data: dict[str, Any],
    mse_data: dict[str, Any],
    title: str,
    default_visible: str = "PBL",
):
    """Build one interactive figure that can toggle PBL, MSE, or both."""
    go = _load_plotly_go()

    _validate_matrix_shapes(
        pbl_data["means"],
        pbl_data["stds"],
        pbl_data["row_labels"],
        pbl_data["col_labels"],
    )
    _validate_matrix_shapes(
        mse_data["means"],
        mse_data["stds"],
        mse_data["row_labels"],
        mse_data["col_labels"],
    )

    if pbl_data["row_labels"] != mse_data["row_labels"]:
        raise ValueError("PBL and MSE row labels do not match.")
    if pbl_data["col_labels"] != mse_data["col_labels"]:
        raise ValueError("PBL and MSE column labels do not match.")

    default_family = default_visible.upper()
    if default_family not in {"PBL", "MSE", "BOTH"}:
        raise ValueError(f"Unsupported default visibility: {default_visible}")

    x_labels = [_display_case(label) for label in pbl_data["col_labels"]]
    cmap = plt.colormaps["tab20"]
    fig = go.Figure()
    trace_metric_types: list[str] = []

    for metric_name, metric_data, opacity in (("PBL", pbl_data, 0.92), ("MSE", mse_data, 0.70)):
        for index, train_case in enumerate(metric_data["row_labels"]):
            customdata = np.array(
                [
                    [
                        _display_case(train_case),
                        float(metric_data["stds"][index, col]),
                        metric_data["metric_key"],
                        metric_name,
                    ]
                    for col in range(len(metric_data["col_labels"]))
                ],
                dtype=object,
            )
            fig.add_bar(
                x=x_labels,
                y=metric_data["means"][index],
                name=f"{metric_name} | {_display_case(train_case)}",
                legendgroup=metric_name,
                marker_color=_to_rgba_string(cmap(index), alpha=opacity),
                error_y={
                    "type": "data",
                    "array": metric_data["stds"][index],
                    "visible": True,
                    "thickness": 1.0,
                    "width": 3,
                },
                offsetgroup=f"{metric_name.lower()}-{index}",
                customdata=customdata,
                hovertemplate=(
                    "Metric family: %{customdata[3]}<br>"
                    "Trained on: %{customdata[0]}<br>"
                    "Metric key: %{customdata[2]}<br>"
                    "Mean: %{y:.6g}<br>"
                    "Std Dev: %{customdata[1]:.6g}<extra></extra>"
                ),
                visible=(default_family == "BOTH" or metric_name == default_family),
            )
            trace_metric_types.append(metric_name)

    pbl_visible = [metric == "PBL" for metric in trace_metric_types]
    mse_visible = [metric == "MSE" for metric in trace_metric_types]
    both_visible = [True] * len(trace_metric_types)

    fig.update_layout(
        title=title if default_family == "BOTH" else f"{title} ({default_family} visible)",
        xaxis_title="Test Case",
        yaxis_title="Mean metric (+/- 1 SD) (log)",
        barmode="group",
        legend_title="Metric | Trained On",
        template="plotly_white",
        width=1650,
        height=700,
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": 1.15,
                "showactive": True,
                "buttons": [
                    {
                        "label": "PBL",
                        "method": "update",
                        "args": [{"visible": pbl_visible}, {"title": f"{title} (PBL visible)"}],
                    },
                    {
                        "label": "MSE",
                        "method": "update",
                        "args": [{"visible": mse_visible}, {"title": f"{title} (MSE visible)"}],
                    },
                    {
                        "label": "Both",
                        "method": "update",
                        "args": [{"visible": both_visible}, {"title": f"{title} (PBL and MSE visible)"}],
                    },
                ],
            }
        ],
    )
    fig.update_yaxes(type="log", showgrid=True, gridcolor="rgba(0, 0, 0, 0.15)")
    fig.show()
    return fig


def write_tabbed_interactive_export(
    tab_specs: list[dict[str, Any]],
    output_html: str | Path,
    page_title: str,
) -> str:
    """Write one self-contained tabbed HTML page for the interactive figures."""
    if not tab_specs:
        raise ValueError("At least one tab specification is required.")

    _load_plotly_go()
    from plotly.offline import get_plotlyjs

    output_path = Path(output_html).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plotly_js = get_plotlyjs()
    button_html_parts = []
    panel_html_parts = []

    for index, spec in enumerate(tab_specs):
        label = str(spec["label"])
        figure = spec["figure"]
        metadata = spec.get("metadata", [])
        tab_id = spec.get("tab_id", f"tab-{index}")
        button_class = "tab-button is-active" if index == 0 else "tab-button"
        panel_class = "tab-panel is-active" if index == 0 else "tab-panel"

        button_html_parts.append(
            f'<button class="{button_class}" data-tab-target="{html.escape(tab_id)}">{html.escape(label)}</button>'
        )

        metadata_html = ""
        if metadata:
            items = "".join(
                f"<li><strong>{html.escape(str(key))}:</strong> {html.escape(str(value))}</li>"
                for key, value in metadata
            )
            metadata_html = f'<ul class="metadata-list">{items}</ul>'

        figure_html = figure.to_html(
            full_html=False,
            include_plotlyjs=False,
            config={"responsive": True},
            div_id=f"plotly-{tab_id}",
        )
        panel_html_parts.append(
            f'<section id="{html.escape(tab_id)}" class="{panel_class}">'
            f'<h2>{html.escape(label)}</h2>'
            f'{metadata_html}'
            f'{figure_html}'
            f'</section>'
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(page_title)}</title>
  <style>
    :root {{
      --bg: #f6f3ee;
      --fg: #182026;
      --muted: #5b6570;
      --line: #d6d0c8;
      --tab-bg: #ebe4d7;
      --tab-active: #1f4f46;
      --tab-active-fg: #ffffff;
      --card: #ffffff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: linear-gradient(180deg, #f2eee7 0%, #faf8f4 100%);
      color: var(--fg);
      font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
    }}
    main {{
      max-width: 1800px;
      margin: 0 auto;
      padding: 2rem;
    }}
    h1 {{
      margin: 0 0 0.5rem 0;
      font-size: 2.1rem;
    }}
    p.lede {{
      margin: 0 0 1.5rem 0;
      color: var(--muted);
      max-width: 70rem;
    }}
    .tab-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-bottom: 1rem;
    }}
    .tab-button {{
      border: 1px solid var(--line);
      background: var(--tab-bg);
      color: var(--fg);
      border-radius: 999px;
      padding: 0.65rem 1rem;
      font: inherit;
      cursor: pointer;
      transition: transform 120ms ease, background 120ms ease;
    }}
    .tab-button:hover {{
      transform: translateY(-1px);
    }}
    .tab-button.is-active {{
      background: var(--tab-active);
      color: var(--tab-active-fg);
      border-color: var(--tab-active);
    }}
    .tab-panel {{
      display: none;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 1.25rem;
      box-shadow: 0 14px 40px rgba(24, 32, 38, 0.08);
    }}
    .tab-panel.is-active {{
      display: block;
    }}
    .tab-panel h2 {{
      margin-top: 0;
      margin-bottom: 0.5rem;
      font-size: 1.35rem;
    }}
    .metadata-list {{
      margin: 0 0 1rem 0;
      padding-left: 1.2rem;
      color: var(--muted);
    }}
  </style>
  <script>{plotly_js}</script>
</head>
<body>
  <main>
    <h1>{html.escape(page_title)}</h1>
    <p class="lede">Each tab below corresponds to one discovered model and training loss family from the April suite. The chart inside each tab is the combined interactive train-vs-test plot with built-in PBL/MSE/Both toggles.</p>
    <div class="tab-row">{''.join(button_html_parts)}</div>
    <div class="tab-panels">{''.join(panel_html_parts)}</div>
  </main>
  <script>
    const buttons = Array.from(document.querySelectorAll('.tab-button'));
    const panels = Array.from(document.querySelectorAll('.tab-panel'));
    function resizePlots(panelId) {{
      if (!window.Plotly) {{
        return;
      }}
      const panel = document.getElementById(panelId);
      if (!panel) {{
        return;
      }}
      panel.querySelectorAll('.plotly-graph-div').forEach((plot) => {{
        window.Plotly.Plots.resize(plot);
      }});
    }}
    function activateTab(targetId) {{
      buttons.forEach((button) => {{
        button.classList.toggle('is-active', button.dataset.tabTarget === targetId);
      }});
      panels.forEach((panel) => {{
        panel.classList.toggle('is-active', panel.id === targetId);
      }});
      window.requestAnimationFrame(() => resizePlots(targetId));
    }}
    buttons.forEach((button) => {{
      button.addEventListener('click', () => activateTab(button.dataset.tabTarget));
    }});
    const initiallyActivePanel = document.querySelector('.tab-panel.is-active');
    if (initiallyActivePanel) {{
      window.requestAnimationFrame(() => resizePlots(initiallyActivePanel.id));
    }}
  </script>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")
    return str(output_path)


def write_interactive_exports(
    figures: dict[str, Any],
    export_dir: str | Path,
    page_title: str,
) -> dict[str, str]:
    """Compatibility helper for older multi-file Plotly exports."""
    export_path = Path(export_dir).resolve()
    export_path.mkdir(parents=True, exist_ok=True)

    written_files: dict[str, str] = {}
    index_items: list[tuple[str, str]] = []

    for filename, figure_info in figures.items():
        html_name = filename if filename.endswith(".html") else f"{filename}.html"
        if isinstance(figure_info, dict):
            if "figure" not in figure_info:
                raise KeyError(f"Figure entry for {filename} is missing figure.")
            figure = figure_info["figure"]
            label = figure_info.get("label", html_name)
        else:
            figure = figure_info
            label = html_name

        output_path = export_path / html_name
        figure.write_html(output_path, full_html=True, include_plotlyjs=True)
        written_files[html_name] = str(output_path)
        index_items.append((label, html_name))

    index_path = export_path / "index.html"
    index_path.write_text(_build_index_html(page_title, index_items), encoding="utf-8")
    written_files["index.html"] = str(index_path)
    return written_files


def _discover_family_roots(run_root: Path) -> list[tuple[str, Path]]:
    children = sorted(path for path in run_root.iterdir() if path.is_dir())
    if not children:
        return []
    if all(TASK_RE.match(path.name) for path in children):
        return [(_normalize_loss_family(run_root.name), run_root)]
    return [(_normalize_loss_family(path.name), path) for path in children]


def _sorted_task_dirs(path: Path) -> list[Path]:
    return sorted((child for child in path.iterdir() if child.is_dir()), key=lambda child: _task_sort_key(child.name))


def _select_best_records_for_model_task(
    model_dir: Path,
    task: str,
    loss_family: str,
    expected_seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    candidate_run_dirs = sorted(
        {
            config_path.parent
            for config_path in model_dir.rglob("config.yaml")
            if (config_path.parent / "summary.json").is_file()
            and (config_path.parent / "val.json").is_file()
        },
        key=lambda path: str(path),
    )
    if not candidate_run_dirs:
        raise FileNotFoundError(f"No run directories were found under {model_dir}")

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    resolved_model_slug: str | None = None

    for run_dir in candidate_run_dirs:
        record = _load_candidate_record(
            run_dir=run_dir,
            task=task,
            loss_family=loss_family,
            model_dir_name=model_dir.name,
            expected_seeds=expected_seeds,
        )
        if resolved_model_slug is None:
            resolved_model_slug = record["model_slug"]
        elif record["model_slug"] != resolved_model_slug:
            raise ValueError(
                f"Inconsistent model slug within {model_dir}: {resolved_model_slug} vs {record['model_slug']}"
            )
        grouped[(record["train_case"], record["seed"])] .append(record)

    selected_records = []
    train_cases = sorted({train_case for train_case, _ in grouped}, key=_case_sort_key)
    for train_case in train_cases:
        seed_keys = tuple(sorted(seed for candidate_train_case, seed in grouped if candidate_train_case == train_case))
        if seed_keys != expected_seeds:
            raise ValueError(
                f"Expected seeds {expected_seeds} for {task} / {resolved_model_slug} / {train_case}, found {seed_keys}"
            )
        for seed in expected_seeds:
            selected_records.append(_pick_best_candidate(grouped[(train_case, seed)]))

    return selected_records


def _pick_best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("Cannot select a best run from an empty candidate list.")

    best_value = min(candidate["selection_metric_value"] for candidate in candidates)
    winners = [
        candidate
        for candidate in candidates
        if math.isclose(candidate["selection_metric_value"], best_value, rel_tol=0.0, abs_tol=1e-12)
    ]
    if len(winners) > 1:
        run_paths = [winner["run_path"] for winner in winners]
        raise ValueError(f"Ambiguous best run tie at {best_value} across {run_paths}")
    return winners[0]


def _load_candidate_record(
    run_dir: Path,
    task: str,
    loss_family: str,
    model_dir_name: str,
    expected_seeds: tuple[int, ...],
) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    val_path = run_dir / "val.json"
    config_path = run_dir / "config.yaml"

    summary = _load_json(summary_path)
    val_history = _load_json(val_path)
    config = _load_yaml(config_path)

    seed = _parse_seed(run_dir.name, expected_seeds)
    config_seed = config.get("functional", {}).get("seed")
    if config_seed is not None and int(config_seed) != seed:
        raise ValueError(
            f"Seed mismatch between directory and config in {run_dir}: {seed} vs {config_seed}"
        )

    dataset_entries = config.get("dataset", {}).get("datasets")
    if not isinstance(dataset_entries, list):
        raise TypeError(f"dataset.datasets must be a list in {config_path}")

    train_entries = [entry for entry in dataset_entries if entry.get("split") == "train"]
    val_entries = [entry for entry in dataset_entries if entry.get("split") == "val"]
    if len(train_entries) != 1:
        raise ValueError(f"Expected exactly one train dataset in {config_path}")
    if len(val_entries) < 2:
        raise ValueError(f"Expected at least one selection val and one grouped test val in {config_path}")

    train_case = train_entries[0].get("case_name")
    if not train_case:
        raise ValueError(f"Missing train case_name in {config_path}")

    selection_dataset_case = val_entries[0].get("case_name")
    test_cases = [entry.get("case_name") for entry in val_entries[1:]]
    if any(not case_name for case_name in test_cases):
        raise ValueError(f"Encountered an empty validation case_name in {config_path}")
    if len(test_cases) != 5:
        raise ValueError(
            f"Expected exactly five grouped test cases in {config_path}, found {len(test_cases)}"
        )

    resolved_val_metrics = _resolve_val_metric_names(config)
    if not resolved_val_metrics:
        raise ValueError(f"No validation losses were resolved from {config_path}")

    selection_metric_key = resolved_val_metrics[0]
    if PBL_METRIC not in resolved_val_metrics:
        raise ValueError(f"{PBL_METRIC} is missing from the resolved validation metrics for {config_path}")
    mse_metric_key = next((metric for metric in resolved_val_metrics if not _is_pbl_metric(metric)), None)
    if mse_metric_key is None:
        raise ValueError(f"No non-PBL comparison metric was resolved from {config_path}")

    required_metrics = {selection_metric_key, PBL_METRIC, mse_metric_key}

    summary_val = summary.get("val")
    if not isinstance(summary_val, list):
        raise TypeError(f"summary.json val is not a list in {summary_path}")
    if len(summary_val) != len(val_entries):
        raise ValueError(
            f"summary.json val length mismatch in {summary_path}: expected {len(val_entries)}, found {len(summary_val)}"
        )

    _validate_metric_entries(summary_val, summary_path, required_metrics)
    _validate_val_history(val_history, val_path, required_metrics, len(val_entries))

    best_point = str(summary.get("best_point", ""))
    if not best_point:
        raise KeyError(f"summary.json is missing best_point: {summary_path}")
    if best_point not in val_history:
        raise KeyError(f"best_point {best_point} was not found in {val_path}")
    if summary_val != val_history[best_point]:
        raise ValueError(f"summary.json val does not match val.json[{best_point}] for {run_dir}")

    selection_metrics = summary_val[0]
    test_case_metrics = {test_cases[index]: summary_val[index + 1] for index in range(len(test_cases))}
    model_slug = _resolve_model_slug(model_dir_name, config)

    return {
        "model_slug": model_slug,
        "loss_family": loss_family,
        "task": task,
        "train_case": train_case,
        "selection_dataset_case": selection_dataset_case,
        "test_cases": test_cases,
        "seed": seed,
        "run_path": str(run_dir.resolve()),
        "run_config_path": str(config_path.resolve()),
        "best_point": best_point,
        "selection_metric_key": selection_metric_key,
        "selection_metric_value": float(selection_metrics[selection_metric_key]),
        "pbl_metric_key": PBL_METRIC,
        "mse_metric_key": mse_metric_key,
        "resolved_val_metrics": resolved_val_metrics,
        "test_case_metrics": test_case_metrics,
    }


def _resolve_val_metric_names(config: dict[str, Any]) -> list[str]:
    val_losses = config.get("optim", {}).get("val_params", {}).get("val_loss")
    if not isinstance(val_losses, list):
        raise TypeError("optim.val_params.val_loss must be a list")
    return [_resolve_loss_metric_name(loss_config) for loss_config in val_losses]


def _resolve_loss_metric_name(loss_config: Any) -> str:
    if isinstance(loss_config, str):
        return _resolve_loss_reference(loss_config, {})
    if not isinstance(loss_config, dict):
        raise TypeError(f"Unsupported loss config type: {type(loss_config)!r}")

    name = loss_config.get("name")
    if not isinstance(name, str):
        raise ValueError(f"Loss config is missing a string name: {loss_config}")

    if name == "combined_loss":
        loss1_label = _resolve_loss_reference(loss_config["loss1"], loss_config.get("inp1", {}))
        loss2_label = _resolve_loss_reference(loss_config["loss2"], loss_config.get("inp2", {}))
        lamb = _format_scalar(loss_config.get("lamb", 1))
        return f"{loss1_label} + {lamb} * {loss2_label}"

    return _resolve_loss_reference(name, loss_config)


def _resolve_loss_reference(loss_name: str, loss_inputs: dict[str, Any]) -> str:
    if loss_name == "universal_power_balance":
        return PBL_METRIC
    if loss_name == "GNNTorchLoss":
        torch_name = loss_inputs.get("torch_nn_name")
        if not torch_name:
            raise ValueError(f"GNNTorchLoss is missing torch_nn_name: {loss_inputs}")
        return str(torch_name)
    if loss_name == "pfn_masked_mse":
        regularize = loss_inputs.get("regularize", True)
        return "Masked MSE, reg." if regularize else "Masked MSE"
    if loss_name == "recycle_loss":
        metric_name = loss_inputs.get("loss_name")
        if not metric_name:
            raise ValueError(f"recycle_loss is missing loss_name: {loss_inputs}")
        return str(metric_name)
    return str(loss_name)


def _resolve_model_slug(model_dir_name: str, config: dict[str, Any]) -> str:
    folder_slug = _normalize_slug(model_dir_name)
    config_slug = _model_slug_from_functional_config(config.get("functional", {}).get("config"))
    model_name_slug = _model_slug_from_model_name(config.get("model", {}).get("name"))

    resolved_slug = folder_slug or config_slug or model_name_slug
    if not resolved_slug:
        raise ValueError(f"Could not determine a model slug from config: {config}")

    mismatches = []
    if folder_slug and config_slug and folder_slug != config_slug:
        mismatches.append(("functional.config", config_slug))
    if folder_slug and model_name_slug and folder_slug != model_name_slug:
        mismatches.append(("model.name", model_name_slug))
    if mismatches:
        mismatch_text = ", ".join(f"{source}={value}" for source, value in mismatches)
        raise ValueError(
            f"Inconsistent model slug across tasks or config sources: folder={folder_slug}, {mismatch_text}"
        )

    return resolved_slug


def _model_slug_from_functional_config(value: Any) -> str | None:
    if value is None:
        return None
    head = str(value).split("/", 1)[0].strip()
    if not head:
        return None

    tokens = [token for token in head.split("_") if token]
    if tokens and TASK_RE.match(tokens[-1]):
        body = tokens[:-1]
        if body and body[-1].lower() in LOSS_FAMILY_SORT_PRIORITY:
            body = body[:-1]
        if not body:
            return None
        return _normalize_slug(body[-1])

    return _normalize_slug(head)


def _model_slug_from_model_name(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip().lower()
    if not raw:
        return None
    if raw.endswith("_pf"):
        raw = raw[:-3]
    return _normalize_slug(raw)


def _validate_val_history(
    val_history: dict[str, Any],
    val_path: Path,
    required_metrics: set[str],
    expected_val_count: int,
) -> None:
    if not isinstance(val_history, dict):
        raise TypeError(f"val.json must be an object mapping epochs to values: {val_path}")
    if not val_history:
        raise ValueError(f"val.json is empty: {val_path}")

    for epoch, entries in val_history.items():
        if not isinstance(entries, list):
            raise TypeError(f"val.json[{epoch}] is not a list in {val_path}")
        if len(entries) != expected_val_count:
            raise ValueError(
                f"val.json[{epoch}] length mismatch in {val_path}: expected {expected_val_count}, found {len(entries)}"
            )
        _validate_metric_entries(entries, val_path, required_metrics)


def _validate_metric_entries(
    entries: list[dict[str, Any]],
    source_path: Path,
    required_metrics: set[str],
) -> None:
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise TypeError(f"Expected a metric dictionary at {source_path} index {index}")
        missing = [metric for metric in required_metrics if metric not in entry]
        if missing:
            raise KeyError(f"Missing required metrics {missing} at {source_path} index {index}")


def _group_signature(records: list[dict[str, Any]]) -> tuple[tuple[str, str], ...]:
    unique_groups = {(record["task"], record["train_case"]) for record in records}
    return tuple(sorted(unique_groups, key=lambda item: (_task_sort_key(item[0]), _case_sort_key(item[1]))))


def _single_unique_value(values: set[str], label: str) -> str:
    if len(values) != 1:
        raise ValueError(f"Expected exactly one {label}, found {sorted(values)}")
    return next(iter(values))


def _task_sort_key(task_name: str) -> tuple[int, str]:
    match = TASK_RE.match(task_name)
    if match is None:
        return (10**9, task_name)
    return (int(match.group("num")), task_name)


def _case_sort_key(case_name: str) -> tuple[int, tuple[int, ...], str]:
    numbers = tuple(int(value) for value in CASE_NUMBER_RE.findall(case_name))
    # Sort by how many cases are combined first so the simple single-case rows
    # appear before the multi-case training combinations, then break ties by the
    # actual numeric case ids.
    return (len(numbers), numbers, case_name)


def _combo_sort_key(combo_key: tuple[str, str]) -> tuple[str, int, str]:
    model_slug, loss_family = combo_key
    return (model_slug, LOSS_FAMILY_SORT_PRIORITY.get(loss_family, 99), loss_family)


def _normalize_loss_family(value: str) -> str:
    normalized = str(value).strip().lower()
    if not normalized:
        raise ValueError("Encountered an empty loss-family label")
    return normalized


def _normalize_slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _format_scalar(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _parse_seed(run_name: str, expected_seeds: tuple[int, ...]) -> int:
    match = SEED_RE.search(run_name)
    if match is None:
        raise ValueError(f"Could not parse a seed from run directory name: {run_name}")
    seed = int(match.group("seed"))
    if seed not in expected_seeds:
        raise ValueError(f"Unexpected seed {seed} in {run_name}; expected one of {expected_seeds}")
    return seed


def _is_pbl_metric(metric_name: str) -> bool:
    return str(metric_name).startswith("PBL")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    return data


def _display_case(case_name: str) -> str:
    stripped = case_name[4:] if case_name.startswith("case") else case_name
    return f"Case {stripped}"


def _validate_matrix_shapes(
    means: np.ndarray,
    stds: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
) -> None:
    if means.shape != stds.shape:
        raise ValueError(f"Mean/std shape mismatch: {means.shape} vs {stds.shape}")
    if means.shape != (len(row_labels), len(col_labels)):
        raise ValueError(
            f"Matrix shape does not match labels: {means.shape} vs {(len(row_labels), len(col_labels))}"
        )


def _load_plotly_go():
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "plotly is required for the interactive April charts. Install plotly and rerun."
        ) from exc
    return go


def _to_rgba_string(
    color_tuple: tuple[float, float, float, float],
    alpha: float | None = None,
) -> str:
    red, green, blue, default_alpha = color_tuple
    actual_alpha = default_alpha if alpha is None else alpha
    return f"rgba({int(red * 255)}, {int(green * 255)}, {int(blue * 255)}, {actual_alpha:.3f})"


def _build_index_html(page_title: str, index_items: list[tuple[str, str]]) -> str:
    items_html = "\n".join(
        f'    <li><a href="{filename}">{label}</a></li>' for label, filename in index_items
    )
    return f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>{page_title}</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 2rem;
      line-height: 1.5;
    }}
  </style>
</head>
<body>
  <h1>{page_title}</h1>
  <ul>
{items_html}
  </ul>
</body>
</html>
"""


__all__ = [
    "DEFAULT_EXPECTED_SEEDS",
    "PBL_METRIC",
    "build_metric_matrices",
    "load_run_family",
    "load_run_suite",
    "plot_grouped_bar_chart",
    "plot_grouped_bar_chart_interactive",
    "plot_metric_toggle_chart",
    "write_interactive_exports",
    "write_tabbed_interactive_export",
]
