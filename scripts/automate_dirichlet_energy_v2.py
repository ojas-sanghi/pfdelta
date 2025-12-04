import os
import shlex
import subprocess
from pathlib import Path


def submit_jobs(base_root: str):
    base_root = Path(base_root).resolve()
    if not base_root.exists():
        raise ValueError(f"Base root {base_root} does not exist")

    sweeps = sorted([d.name for d in base_root.iterdir() if d.is_dir()])

    # Run from the repo root (adjust if needed)
    repo_root = Path.cwd()

    for sweep in sweeps:
        sweep_root = base_root / sweep

        # The command we actually want to run on the node
        inner = f"uv run scripts/extract_dirichlet_energy.py --root {shlex.quote(str(sweep_root))}"

        cmd = [
            "sbatch",
            "--job-name=dirichlet",
            "--time=1:00:00",
            "-p",
            "cpu",
            "--chdir",
            str(repo_root),  # ensure relative paths like scripts/... resolve
            "--wrap",
            f"bash -lc {shlex.quote(inner)}",  # login shell so PATH/env is loaded
            "--output",
            f"{sweep_root}/dirichlet_%j.out",
        ]

        print(" ".join(shlex.quote(c) for c in cmd), end="\n\n")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    # run_path = "runs/gen_msrp_fe/run1/task33/graphconv/"
    run_path = "runs/gen_msrp_fe/run1/task33/powerflownet/"
    submit_jobs(run_path)
