"""
Find config .yaml and .sh files to requeue for failed training runs.

A "failed" run = a run folder that has config.yaml and out.txt but NO summary.json
(i.e., training never completed successfully).

Usage:
    uv run scripts/find_failed_runs.py --runs-dir runs/gen_apr0326
    uv run scripts/find_failed_runs.py --runs-dir runs/gen_pbl_apr0326
    uv run scripts/find_failed_runs.py --runs-dir runs/gen_apr0326 --print-sh
"""

import os
import sys
import argparse
import yaml
from pathlib import Path


def find_failed_run_locations(runs_dir: Path) -> set[str]:
    """
    Walk the runs directory tree. A leaf folder is a "run" if it contains
    out.txt. It's "failed" if it does NOT contain summary.json.
    Returns a set of relative run paths (matching functional.run_location).
    """
    failed = set()

    for root, dirs, files in os.walk(runs_dir):
        if "out.txt" in files and "summary.json" not in files:
            rel = Path(root).relative_to(runs_dir.parent)
            failed.add(str(rel))
            dirs.clear()

    return failed


def find_configs_for_failed_runs(configs_dir: Path, failed_locations: set[str], runs_dir_name: str):
    """
    Walk core/configs, parse every .yaml, check functional.run_name.
    Only searches config folders whose name starts with runs_dir_name to avoid
    cross-contamination (e.g. gen_apr0326 vs gen_pbl_apr0326).
    """
    matches = []  # list of (yaml_path, sh_path, run_name)

    for yaml_path in sorted(configs_dir.rglob("*.yaml")):
        if "archive" in str(yaml_path):
            continue

        # Only match configs from the corresponding config folder prefix
        if not yaml_path.parent.name.startswith(runs_dir_name):
            continue

        try:
            with open(yaml_path) as f:
                cfg = yaml.safe_load(f)
        except Exception as e:
            print(f"  [WARN] Could not parse {yaml_path}: {e}", file=sys.stderr)
            continue

        run_name = cfg.get("functional", {}).get("run_name", "")
        if not run_name:
            continue

        run_name_norm = run_name.replace("\\", "/")

        for failed in failed_locations:
            if run_name_norm in failed.replace("\\", "/"):
                sh_path = yaml_path.with_suffix(".sh")
                matches.append((yaml_path, sh_path if sh_path.exists() else None, run_name_norm))
                break

    return matches


def main():
    parser = argparse.ArgumentParser(description="Find configs for failed training runs.")
    parser.add_argument("--runs-dir", default="runs", help="Path to runs/ directory")
    parser.add_argument("--configs-dir", default="core/configs", help="Path to core/configs/ directory")
    parser.add_argument("--print-sh", action="store_true", help="Print only the bash commands to requeue")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    configs_dir = Path(args.configs_dir).resolve()

    if not runs_dir.exists():
        print(f"ERROR: runs dir not found: {runs_dir}", file=sys.stderr)
        sys.exit(1)
    if not configs_dir.exists():
        print(f"ERROR: configs dir not found: {configs_dir}", file=sys.stderr)
        sys.exit(1)

    runs_dir_name = runs_dir.name  # e.g. "gen_apr0326"

    print(f"Scanning for failed runs in: {runs_dir}")
    failed_locations = find_failed_run_locations(runs_dir)
    print(f"  Found {len(failed_locations)} failed run(s).\n")

    if not failed_locations:
        print("No failed runs found. Nothing to requeue.")
        return

    if not args.print_sh:
        print("Failed run locations:")
        for loc in sorted(failed_locations):
            print(f"  {loc}")
        print()

    print(f"Searching configs in: {configs_dir} (prefix: {runs_dir_name})")
    matches = find_configs_for_failed_runs(configs_dir, failed_locations, runs_dir_name)
    print(f"  Matched {len(matches)} config file(s).\n")

    if not matches:
        print("WARNING: No matching config files found.")
        print("Check that functional.run_name in your yamls matches the runs/ folder structure.")
        return

    if args.print_sh:
        print("# --- Commands to requeue failed runs ---")
        for yaml_path, sh_path, run_name in matches:
            if sh_path:
                print(f"sbatch {sh_path}")
            else:
                print(f"# WARNING: no .sh found for {yaml_path}")
    else:
        print("Configs to rerun:")
        print(f"{'Run Name':<70}  {'Config YAML':<50}  {'Shell Script'}")
        print("-" * 160)
        for yaml_path, sh_path, run_name in matches:
            sh_str = str(sh_path) if sh_path else "NOT FOUND"
            print(f"{run_name:<70}  {str(yaml_path):<50}  {sh_str}")

        matched_run_names = {r for _, _, r in matches}
        unmatched = {
            f for f in failed_locations
            if not any(r in f.replace("\\", "/") for r in matched_run_names)
        }
        if unmatched:
            print(f"\nWARNING: {len(unmatched)} failed run(s) had no matching config yaml:")
            for loc in sorted(unmatched):
                print(f"  {loc}")
        else:
            print(f"\nAll {len(failed_locations)} failed runs matched successfully.")


if __name__ == "__main__":
    main()