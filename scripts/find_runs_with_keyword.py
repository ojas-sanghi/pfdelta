"""
List run names whose out.txt contains a given keyword.

Examples:
    python3 scripts/find_runs_with_keyword.py
    python3 scripts/find_runs_with_keyword.py --runs-dir runs/gen_apr0326
    python3 scripts/find_runs_with_keyword.py --keyword "Early stop triggered"
    python3 scripts/find_runs_with_keyword.py --print-paths
"""

import argparse
import re
import signal
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

EPOCH_RE = re.compile(r"Epoch\s+(\d+)/(\d+)\s+done")
MODEL_MAX_EPOCHS = {
    "canos": 200,
    "powerflownet": 100,
    "graphconv": 100,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find run directories whose out.txt contains a keyword."
    )
    parser.add_argument(
        "--runs-dir",
        default="runs/gen_apr0326",
        help="Root directory containing nested run folders.",
    )
    parser.add_argument(
        "--keyword",
        default="Early stop triggered",
        help="Keyword to search for inside each out.txt file.",
    )
    parser.add_argument(
        "--print-paths",
        action="store_true",
        help="Print run paths relative to the runs dir instead of only the run name.",
    )
    return parser.parse_args()


def infer_model(run_dir):
    parts = run_dir.parts
    for model in MODEL_MAX_EPOCHS:
        if model in parts:
            return model
    return "unknown"


def extract_run_info(out_path, keyword):
    last_epoch = None  # type: Optional[int]
    keyword_seen = False

    try:
        with out_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                match = EPOCH_RE.search(line)
                if match:
                    last_epoch = int(match.group(1))

                if keyword in line:
                    keyword_seen = True
    except OSError as exc:
        print("[WARN] Could not read {}: {}".format(out_path, exc), file=sys.stderr)
        return None, False

    return last_epoch, keyword_seen


def collect_runs(runs_dir, keyword):
    all_runs = []  # type: List[Tuple[Path, str, Optional[int], Optional[int], bool]]
    matches = []  # type: List[Tuple[Path, str, Optional[int], Optional[int], bool]]
    checked = 0

    for out_file in sorted(runs_dir.rglob("out.txt")):
        checked += 1
        last_epoch, keyword_seen = extract_run_info(out_file, keyword)

        run_dir = out_file.parent
        model = infer_model(run_dir)
        max_epochs = MODEL_MAX_EPOCHS.get(model)
        record = (run_dir, model, max_epochs, last_epoch, keyword_seen)
        all_runs.append(record)

        if keyword_seen:
            matches.append(record)

    return checked, all_runs, matches


def print_grouped_summary(all_runs):
    counts = Counter()

    for _, model, _, last_epoch, _ in all_runs:
        counts[(model, last_epoch)] += 1

    print("\nGrouped Summary (All Checked Runs)")
    print("----------------------------------")

    grouped_total = 0
    for (model, last_epoch), run_count in sorted(
        counts.items(), key=lambda item: (item[0][0], item[0][1] is None, item[0][1])
    ):
        epoch_label = "?" if last_epoch is None else str(last_epoch)
        print("{} | {} epochs | {} runs".format(model, epoch_label, run_count))
        grouped_total += run_count

    print("Total grouped runs: {}".format(grouped_total))


def main():
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    args = parse_args()
    runs_dir = Path(args.runs_dir).resolve()

    if not runs_dir.exists():
        print("ERROR: runs dir not found: {}".format(runs_dir), file=sys.stderr)
        return 1

    checked, all_runs, matches = collect_runs(runs_dir, args.keyword)

    header = "{:<14} {:<10} {:<11} {}".format(
        "model", "max_epochs", "stop_epoch", "run"
    )
    print(header)
    print("-" * len(header))

    for run_dir, model, max_epochs, last_epoch, _ in matches:
        run_label = run_dir.relative_to(runs_dir) if args.print_paths else run_dir.name
        max_epochs_label = max_epochs if max_epochs is not None else "?"
        stop_epoch_label = last_epoch if last_epoch is not None else "?"
        print(
            "{:<14} {:<10} {:<11} {}".format(
                model, max_epochs_label, stop_epoch_label, run_label
            )
        )

    print_grouped_summary(all_runs)

    print("\nChecked {} run(s).".format(checked), file=sys.stderr)
    print("Keyword appeared in {} run(s).".format(len(matches)), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
