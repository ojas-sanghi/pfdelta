#!/usr/bin/env python3
"""Deterministic best-run extraction for the Feb 01 2026 training sweep.

Purpose:
This script is the "selection and extraction" stage of the pipeline. It scans
all experiment folders, chooses the best run *per seed* for each
`(task, model, selection_case)` group, and writes normalized CSV artifacts used
by downstream plotting (`visualize_best_epoch.py`) and audit workflows.

Notebook parity:
The group definitions, case-index selection, and seed-handling behavior are
kept consistent with:
- `notebooks/ojas_feb0126_test_case_viz_interactive.ipynb`
- `notebooks/ojas_feb0126_find_best_run.py`

Expected input shape:
- Runs live under `--runs-root` (default `runs/gen_feb0126`)
- Each candidate run directory is expected to include:
  - `summary.json` with `summary["val"][case_index][error_key]`
  - `val.json` with per-epoch validation metrics
- Seed is parsed from run directory name via `seed=<int>`

Selection logic summary:
1. Build all selection groups across tasks/models/cases.
2. Discover candidate run folders under each group.
3. Filter candidates to expected seeds (42, 43, 44) with valid files/metrics.
4. Select one best run per seed using:
   - primary key: minimum selected validation error
   - tie-break: lexicographically smaller run path
5. Extract full epoch-vs-metric curves from each selected run.

Output artifacts:
- `selected_runs_detailed.csv`
  One row per selected `(task, model, selection_case, seed)` run with summary
  stats and provenance metadata.
- `selected_runs_curve_points.csv`
  One row per selected curve point `(task, model, selection_case, seed, epoch)`.
  This is the primary input to `visualize_best_epoch.py`.
- `selected_group_summary.csv`
  Aggregate seed-level selection stats per `(task, model, selection_case)`.

Determinism guarantees:
- Recursive directory traversal is sorted.
- Duplicate discovered run dirs are de-duplicated in stable order.
- Tie-breaking is explicit.
- CSV row order is explicitly sorted before writing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Tuple

# Model families evaluated in the Feb 01 2026 experiments.
MODELS: Tuple[str, ...] = ("graphconv", "powerflownet")
# Task 31 single-case training groups.
TASK31_CASES: Tuple[str, ...] = ("case14", "case30", "case57", "case118", "case500")
# Task 34 multi-case training groups.
TASK34_CASES: Tuple[str, ...] = (
    "case14_30_118",
    "case30_57_118",
    "case30_57_500",
    "case57_118_500",
)
# Notebook expects three seeds and maps them to array slots [0, 1, 2].
SEED_TO_SLOT: Dict[int, int] = {42: 0, 43: 1, 44: 2}
EXPECTED_SEEDS: Tuple[int, ...] = (42, 43, 44)
SEED_RE = re.compile(r"seed=(\d+)")


@dataclass(frozen=True)
class SelectionGroup:
    """One selection unit that mirrors a single `find_best_run(...)` call in the notebook.

    Attributes:
        task: Task identifier (`task31`, `task32`, `task33`, `task34`).
        model: Model family (`graphconv` or `powerflownet`).
        selection_case: Case grouping used to choose the metric index and folder.
        folder: Root folder under which candidate run directories are searched.
    """

    task: str
    model: str
    selection_case: str
    folder: Path


@dataclass(frozen=True)
class CandidateRun:
    """Parsed candidate run with only fields needed for deterministic ranking.

    Attributes:
        seed: Seed parsed from the run folder name.
        seed_slot: Slot mapping used by original notebook logic (42->0, 43->1, 44->2).
        run_path: Path to the concrete run directory.
        summary_path: Path to `summary.json`.
        val_path: Path to `val.json`.
        selected_error: Metric value extracted from `summary.json` for this selection.
        summary_best_point_raw: Raw `best_point` value as stored in summary.
    """

    seed: int
    seed_slot: int
    run_path: Path
    summary_path: Path
    val_path: Path
    selected_error: float
    summary_best_point_raw: str


def parse_args() -> argparse.Namespace:
    """Parse command-line options for run selection.

    Returns:
        Parsed argument namespace with:
        - `runs_root`: root folder containing experiment runs.
        - `output_dir`: target folder for CSV artifacts.
        - `error_key`: validation metric used for selecting best run per seed.

    Notes:
        `error_key` must exist in both:
        - `summary["val"][case_index]`
        - `val.json[epoch][case_index]`
        for a candidate/point to be considered valid.
    """
    here = Path(__file__).resolve().parent
    default_output_dir = here / "outputs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs/gen_feb0126"),
        help="Root run directory (default: runs/gen_feb0126)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"Directory for generated CSV files (default: {default_output_dir})",
    )
    parser.add_argument(
        "--error-key",
        type=str,
        default="PBL Mean",
        help="Validation metric key used for best-run selection (default: PBL Mean)",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    """Load JSON content from `path`.

    The function intentionally does not swallow decode errors because malformed
    JSON should fail fast during deterministic artifact generation.
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def case_index_for_selection(task: str, selection_case: str) -> int:
    """Return which entry in `summary['val']` / `val.json[epoch]` to evaluate.

    Behavior intentionally mirrors notebook logic:
    - `task31`: index is `TASK31_CASES.index(selection_case) + 1`
      because index 0 is the train-set validation entry.
    - `task32`, `task33`, `task34`: index 0.
    """
    if task == "task31":
        return TASK31_CASES.index(selection_case) + 1
    return 0


def parse_seed_from_path(run_path: Path) -> Optional[int]:
    """Extract integer seed from run directory name (e.g. `seed=42_...`)."""
    match = SEED_RE.search(run_path.name)
    if match is None:
        return None
    return int(match.group(1))


def deterministic_run_dirs(folder: Path) -> List[Path]:
    """Return unique run directories in deterministic order.

    A run directory is recognized by the presence of `summary.json`.
    The list is sorted lexicographically by path, then deduplicated while
    preserving order.

    Why this matters:
    `Path.rglob(...)` can produce non-deterministic traversal order across
    environments/filesystems. Explicit sorting and stable de-duplication keep
    downstream CSV diffs reproducible.
    """
    summary_paths = sorted(folder.rglob("summary.json"), key=lambda p: p.as_posix())
    run_dirs = [p.parent for p in summary_paths]

    # Keep deterministic order and uniqueness.
    seen = set()
    unique: List[Path] = []
    for run_dir in run_dirs:
        run_key = run_dir.as_posix()
        if run_key not in seen:
            seen.add(run_key)
            unique.append(run_dir)
    return unique


def build_selection_groups(runs_root: Path) -> List[SelectionGroup]:
    """Build all selection groups used by the notebook.

    Per model, groups are:
    - task31: 5 single-case groups
    - task32: 1 grouped case
    - task33: 1 grouped case
    - task34: 4 grouped cases

    Total groups = 11 per model, 22 across two models.
    """
    groups: List[SelectionGroup] = []
    for model in MODELS:
        for case in TASK31_CASES:
            groups.append(
                SelectionGroup(
                    task="task31",
                    model=model,
                    selection_case=case,
                    folder=runs_root / "task31" / model / case,
                )
            )
        groups.append(
            SelectionGroup(
                task="task32",
                model=model,
                selection_case="case14_30_57",
                folder=runs_root / "task32" / model,
            )
        )
        groups.append(
            SelectionGroup(
                task="task33",
                model=model,
                selection_case="case118_500",
                folder=runs_root / "task33" / model,
            )
        )
        for case in TASK34_CASES:
            groups.append(
                SelectionGroup(
                    task="task34",
                    model=model,
                    selection_case=case,
                    folder=runs_root / "task34" / model / case,
                )
            )
    return sorted(groups, key=lambda g: (g.model, g.task, g.selection_case))


def candidate_from_run(
    run_path: Path,
    task: str,
    selection_case: str,
    error_key: str,
) -> Optional[CandidateRun]:
    """Parse a run directory into a `CandidateRun` if it is valid for selection.

    Validation/filtering:
    - seed must be one of expected seeds.
    - both `summary.json` and `val.json` must exist.
    - selected metric (`error_key`) must be readable from summary at the
      computed case index.

    Returns:
        `CandidateRun` on success, otherwise `None`.

    Important:
        This function only validates fields needed for ranking and extraction.
        It does not guarantee that every epoch in `val.json` is valid; per-point
        filtering is handled later in `curve_points_for_candidate(...)`.
    """
    seed = parse_seed_from_path(run_path)
    if seed not in SEED_TO_SLOT:
        return None

    summary_path = run_path / "summary.json"
    val_path = run_path / "val.json"
    if not summary_path.exists() or not val_path.exists():
        return None

    summary = load_json(summary_path)
    case_index = case_index_for_selection(task, selection_case)
    try:
        selected_error = float(summary["val"][case_index][error_key])
    except (KeyError, IndexError, TypeError, ValueError):
        return None

    best_point_raw = str(summary.get("best_point", ""))

    return CandidateRun(
        seed=seed,
        seed_slot=SEED_TO_SLOT[seed],
        run_path=run_path,
        summary_path=summary_path,
        val_path=val_path,
        selected_error=selected_error,
        summary_best_point_raw=best_point_raw,
    )


def choose_best_per_seed(candidates: Iterable[CandidateRun]) -> Dict[int, CandidateRun]:
    """Choose one best run per seed using deterministic tie-breaking.

    Primary criterion: lowest `selected_error`.
    Tie-breaker: lexicographically smaller run path.

    Returns:
        Mapping from seed to chosen candidate; missing seeds are omitted.

    Example:
        If seed 43 has no valid candidates for a group, seed 43 is simply
        absent from the returned dict. The group can still contribute rows for
        seeds 42/44.
    """
    by_seed: Dict[int, List[CandidateRun]] = {seed: [] for seed in EXPECTED_SEEDS}
    for candidate in candidates:
        by_seed[candidate.seed].append(candidate)

    selected: Dict[int, CandidateRun] = {}
    for seed in EXPECTED_SEEDS:
        seed_candidates = by_seed[seed]
        if not seed_candidates:
            continue
        seed_candidates_sorted = sorted(
            seed_candidates,
            key=lambda c: (c.selected_error, c.run_path.as_posix()),
        )
        selected[seed] = seed_candidates_sorted[0]
    return selected


def parse_best_point(raw: str) -> Optional[int]:
    """Parse `summary.best_point` as int, returning `None` when unavailable/invalid."""
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def curve_points_for_candidate(
    candidate: CandidateRun,
    task: str,
    selection_case: str,
    error_key: str,
) -> List[Tuple[int, float]]:
    """Extract an epoch->metric curve for a selected candidate run.

    Reads `candidate.val_path` and keeps entries where:
    - epoch key is integer-like.
    - selected case index exists.
    - metric `error_key` is present and numeric.

    Returns:
        Sorted list of `(epoch, metric_value)` pairs.

    Epoch filtering behavior:
        Invalid epochs are skipped (non-integer keys, missing case index, or
        non-numeric metric values). This keeps extraction robust to partial or
        irregular logging output.
    """
    val_json = load_json(candidate.val_path)
    case_index = case_index_for_selection(task, selection_case)

    points: List[Tuple[int, float]] = []
    for epoch_str, values in val_json.items():
        try:
            epoch = int(epoch_str)
        except (TypeError, ValueError):
            continue

        try:
            metric_value = float(values[case_index][error_key])
        except (KeyError, IndexError, TypeError, ValueError):
            continue

        points.append((epoch, metric_value))

    return sorted(points, key=lambda pair: pair[0])


def curve_summary(points: List[Tuple[int, float]]) -> Dict[str, float]:
    """Compute compact curve statistics used in the detailed CSV.

    For non-empty input, returns:
    - number of points
    - first/mid/last epoch+value
    - best (minimum value) epoch+value

    For empty input, numeric fields are set to `NaN` (and n_points=0).

    These summary values are convenience columns for sorting/auditing in CSV;
    they are not used by selection logic.
    """
    if not points:
        return {
            "curve_n_points": 0,
            "curve_first_epoch": math.nan,
            "curve_first_value": math.nan,
            "curve_mid_epoch": math.nan,
            "curve_mid_value": math.nan,
            "curve_last_epoch": math.nan,
            "curve_last_value": math.nan,
            "curve_best_epoch": math.nan,
            "curve_best_value": math.nan,
        }

    first_epoch, first_value = points[0]
    mid_epoch, mid_value = points[len(points) // 2]
    last_epoch, last_value = points[-1]
    best_epoch, best_value = min(points, key=lambda pair: pair[1])

    return {
        "curve_n_points": len(points),
        "curve_first_epoch": first_epoch,
        "curve_first_value": first_value,
        "curve_mid_epoch": mid_epoch,
        "curve_mid_value": mid_value,
        "curve_last_epoch": last_epoch,
        "curve_last_value": last_value,
        "curve_best_epoch": best_epoch,
        "curve_best_value": best_value,
    }


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    """Write rows to CSV with explicit header order.

    Header order is fixed so artifact diffs stay stable across runs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    """Run end-to-end analysis and write deterministic CSV artifacts.

    Output files:
    - `selected_runs_detailed.csv`: one row per selected run (group + seed).
    - `selected_runs_curve_points.csv`: one row per (selected run, epoch) point.
    - `selected_group_summary.csv`: aggregate stats per (task, model, selection_case).

    Processing outline:
    1. Build all target groups.
    2. Discover + validate candidate runs.
    3. Select one run per seed.
    4. Extract detailed rows and full curve-point rows.
    5. Compute per-group aggregate summary rows.
    6. Sort rows deterministically and write CSV artifacts.
    """
    args = parse_args()
    runs_root = args.runs_root.resolve()
    output_dir = args.output_dir.resolve()
    error_key = args.error_key

    groups = build_selection_groups(runs_root)

    detailed_rows: List[dict] = []
    curve_rows: List[dict] = []
    group_rows: List[dict] = []

    for group in groups:
        if not group.folder.exists():
            continue

        # Collect all valid run candidates under this group.
        candidates: List[CandidateRun] = []
        for run_dir in deterministic_run_dirs(group.folder):
            candidate = candidate_from_run(
                run_path=run_dir,
                task=group.task,
                selection_case=group.selection_case,
                error_key=error_key,
            )
            if candidate is not None:
                candidates.append(candidate)

        selected_by_seed = choose_best_per_seed(candidates)

        selected_errors: List[float] = []
        selected_best_points: List[int] = []

        for seed in EXPECTED_SEEDS:
            candidate = selected_by_seed.get(seed)
            if candidate is None:
                continue

            points = curve_points_for_candidate(candidate, group.task, group.selection_case, error_key)
            summary = curve_summary(points)
            best_point = parse_best_point(candidate.summary_best_point_raw)
            if best_point is not None:
                selected_best_points.append(best_point)

            selected_errors.append(candidate.selected_error)

            line_label = (
                f"{group.task}|{group.model}|{group.selection_case}|"
                f"seed={seed}|bp={best_point}|err={candidate.selected_error:.6f}"
            )

            detailed_row = {
                "task": group.task,
                "model": group.model,
                "selection_case": group.selection_case,
                "seed": seed,
                "seed_slot": candidate.seed_slot,
                "error_key": error_key,
                "selected_error": candidate.selected_error,
                "summary_best_point_raw": candidate.summary_best_point_raw,
                "summary_best_point": "" if best_point is None else best_point,
                "run_path": candidate.run_path.as_posix(),
                "line_label": line_label,
                **summary,
            }
            detailed_rows.append(detailed_row)

            # Store full curve points for downstream plotting and analysis.
            for epoch, value in points:
                curve_rows.append(
                    {
                        "task": group.task,
                        "model": group.model,
                        "selection_case": group.selection_case,
                        "seed": seed,
                        "seed_slot": candidate.seed_slot,
                        "error_key": error_key,
                        "selected_error": candidate.selected_error,
                        "summary_best_point": "" if best_point is None else best_point,
                        "line_label": line_label,
                        "run_path": candidate.run_path.as_posix(),
                        "epoch": epoch,
                        "metric_value": value,
                    }
                )

        if selected_errors:
            group_rows.append(
                {
                    "task": group.task,
                    "model": group.model,
                    "selection_case": group.selection_case,
                    "error_key": error_key,
                    "n_selected_seeds": len(selected_errors),
                    "selected_error_mean": mean(selected_errors),
                    "selected_error_std": pstdev(selected_errors) if len(selected_errors) > 1 else 0.0,
                    "selected_error_min": min(selected_errors),
                    "selected_error_max": max(selected_errors),
                    "summary_best_point_min": min(selected_best_points) if selected_best_points else "",
                    "summary_best_point_max": max(selected_best_points) if selected_best_points else "",
                }
            )

    # Deterministic output ordering for reproducible CSV diffs.
    detailed_rows.sort(
        key=lambda row: (
            row["model"],
            row["task"],
            row["selection_case"],
            int(row["seed"]),
            row["run_path"],
        )
    )
    curve_rows.sort(
        key=lambda row: (
            row["model"],
            row["task"],
            row["selection_case"],
            int(row["seed"]),
            int(row["epoch"]),
            row["run_path"],
        )
    )
    group_rows.sort(key=lambda row: (row["model"], row["task"], row["selection_case"]))

    # Rank globally by selected error (smaller is better), with fixed tie-breakers.
    for rank, row in enumerate(
        sorted(
            detailed_rows,
            key=lambda r: (r["selected_error"], r["model"], r["task"], r["selection_case"], r["seed"]),
        ),
        start=1,
    ):
        row["global_error_rank"] = rank

    # Re-sort after rank assignment so file ordering stays grouped by model/task/case/seed.
    detailed_rows.sort(
        key=lambda row: (
            row["model"],
            row["task"],
            row["selection_case"],
            int(row["seed"]),
            row["run_path"],
        )
    )

    detailed_csv = output_dir / "selected_runs_detailed.csv"
    curve_csv = output_dir / "selected_runs_curve_points.csv"
    groups_csv = output_dir / "selected_group_summary.csv"

    write_csv(
        detailed_csv,
        detailed_rows,
        fieldnames=[
            "task",
            "model",
            "selection_case",
            "seed",
            "seed_slot",
            "error_key",
            "selected_error",
            "summary_best_point_raw",
            "summary_best_point",
            "curve_n_points",
            "curve_first_epoch",
            "curve_first_value",
            "curve_mid_epoch",
            "curve_mid_value",
            "curve_last_epoch",
            "curve_last_value",
            "curve_best_epoch",
            "curve_best_value",
            "global_error_rank",
            "line_label",
            "run_path",
        ],
    )

    write_csv(
        curve_csv,
        curve_rows,
        fieldnames=[
            "task",
            "model",
            "selection_case",
            "seed",
            "seed_slot",
            "error_key",
            "selected_error",
            "summary_best_point",
            "line_label",
            "run_path",
            "epoch",
            "metric_value",
        ],
    )

    write_csv(
        groups_csv,
        group_rows,
        fieldnames=[
            "task",
            "model",
            "selection_case",
            "error_key",
            "n_selected_seeds",
            "selected_error_mean",
            "selected_error_std",
            "selected_error_min",
            "selected_error_max",
            "summary_best_point_min",
            "summary_best_point_max",
        ],
    )

    total_selected = len(detailed_rows)
    total_points = len(curve_rows)

    print(f"runs_root={runs_root}")
    print(f"error_key={error_key}")
    print(f"selected_runs={total_selected}")
    print(f"curve_points={total_points}")
    print(f"wrote={detailed_csv}")
    print(f"wrote={curve_csv}")
    print(f"wrote={groups_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
