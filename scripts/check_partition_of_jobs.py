"""
Scan all .out files in the current directory, determine which jobs failed vs succeeded,
run sacct to get partition info, and report grouped results.

Usage:
    uv run scripts/check_partition_of_jobs.py
    uv run scripts/check_partition_of_jobs.py --out-dir /path/to/outfiles
"""

import re
import subprocess
import argparse
from pathlib import Path
from collections import defaultdict


FAILURE_SIGNALS = [
    # "Traceback (most recent call last)",
    # "Error:",
    # "error:",
    # "CUDA error",
    # "RuntimeError",
    # "AssertionError",
    # "OutOfMemoryError",
    # "FAILED",
    # "slurmstepd: error",
    # "DUE TO TIME LIMIT",
    "torch.AcceleratorError",
    "Device being used: cpu"
]

SUCCESS_SIGNALS = [
    # "Training complete",
    # "Finished training",
    # "summary.json",
    # "Best model saved",
    # "Training finished",
    "Results Summary"
]


def classify_out_file(path: Path) -> str:
    """Returns 'failed', 'succeeded', or 'unknown'."""
    try:
        text = path.read_text(errors="replace")
    except Exception:
        return "unknown"

    for sig in FAILURE_SIGNALS:
        if sig in text:
            return "failed"
    for sig in SUCCESS_SIGNALS:
        if sig in text:
            return "succeeded"
    return "unknown"


def get_sacct_info(job_id: str) -> dict:
    """Run sacct -j <job_id> and extract partition and state."""
    try:
        result = subprocess.run(
            ["sacct", "-j", job_id, "--format=JobID,JobName,Partition,State", "--noheader"],
            capture_output=True, text=True, timeout=10
        )
        lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
        # First line is the main job entry (not .batch or .extern)
        for line in lines:
            parts = line.split()
            if len(parts) >= 4 and "." not in parts[0]:
                partition = parts[2]
                state = parts[3]
                # Normalize partition name
                if "preem" in partition.lower():
                    partition_label = "mit_preempt"
                elif "normal" in partition.lower() or "norma" in partition.lower():
                    partition_label = "mit_normal"
                else:
                    partition_label = partition
                return {"partition": partition_label, "state": state}
    except Exception as e:
        return {"partition": "unknown", "state": "unknown"}
    return {"partition": "unknown", "state": "unknown"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=".", help="Directory containing .out files")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_files = sorted(out_dir.glob("*.out"))

    if not out_files:
        print(f"No .out files found in {out_dir}")
        return

    print(f"Found {len(out_files)} .out files in {out_dir}\n")

    # job_id -> {filename, classification, partition, sacct_state}
    results = {}

    for f in out_files:
        # Extract job ID: last numeric segment before .out
        match = re.search(r'_(\d+)\.out$', f.name)
        if not match:
            print(f"  [SKIP] Can't parse job ID from: {f.name}")
            continue
        job_id = match.group(1)

        classification = classify_out_file(f)
        sacct = get_sacct_info(job_id)

        results[job_id] = {
            "filename": f.name,
            "classification": classification,
            "partition": sacct["partition"],
            "sacct_state": sacct["state"],
        }

        print(f"  {f.name:<40}  job={job_id}  [{classification}]  partition={sacct['partition']}  sacct={sacct['state']}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Group by classification -> partition -> job_ids
    groups = defaultdict(lambda: defaultdict(list))
    for job_id, info in results.items():
        groups[info["classification"]][info["partition"]].append(job_id)

    for classification in ["failed", "succeeded", "unknown"]:
        if classification not in groups:
            continue
        partition_map = groups[classification]
        total = sum(len(v) for v in partition_map.values())
        print(f"\n{'=' * 30}")
        print(f"  {classification.upper()} ({total} jobs)")
        print(f"{'=' * 30}")
        for partition, job_ids in sorted(partition_map.items()):
            print(f"\n  Partition: {partition} ({len(job_ids)} jobs)")
            for jid in sorted(job_ids):
                fname = results[jid]["filename"]
                print(f"    job {jid}  ({fname})")


if __name__ == "__main__":
    main()