import os
import subprocess
import sys
from pathlib import Path


def submit_jobs(base_root: str):
    base_root: Path = Path(base_root).resolve()

    if not base_root.exists():
        raise ValueError(f"Base root {base_root} does not exist")

    sweeps = [d.name for d in base_root.iterdir() if d.is_dir()]
    sweeps.sort()

    # base_out = Path("runs") / base_root.relative_to(base_root.parts[0])

    for sweep in sweeps[0:1]:
        sweep_root: Path = base_root / sweep
        # sweep_out = base_out / sweep
        # sweep_out.mkdir(parents=True, exist_ok=True)

        cmd = [
            "sbatch",
            "--job-name=dirichlet",
            "--time=1:00:00",
            "-p",
            "cpu",
            "--wrap",
            f'"uv run scripts/extract_dirichlet_energy.py --root {sweep_root}"',
            "--output",
            f"{sweep_root}/dirichlet_%j.out",
        ]

        print(" ".join(cmd), end="\n\n")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    run_path = "runs/gen_msrp_fe/run1/task3.1/case14/graphconv/"

    submit_jobs(run_path)

"""
for documentation, alternatives before this script was made:

sbatch --job-name="dirichlet" --time=1:00:00 -p cpu --wrap="uv run scripts/extract_dirichlet_energy.py --root gen_msrp_fe/run1/task3.1/case14/graphconv/sweep_case_14_layers_5_hidden_256_lr_1e-3_epochs_100_2_250917_225934" --output=runs/gen_msrp_fe/run1/task3.1/case14/graphconv/sweep_case_14_layers_5_hidden_256_lr_1e-3_epochs_100_2_250917_225934/dirichlet_%j.out

# or
debugcpu
uv run scripts/extract_dirichlet_energy.py --root gen_msrp_fe/run1/task3.1/case14/graphconv/sweep_case_14_layers_5_hidden_256_lr_1e-3_epochs_100_2_250917_225934 >> runs/gen_msrp_fe/run1/task3.1/case14/graphconv/sweep_case_14_layers_5_hidden_256_lr_1e-3_epochs_100_2_250917_225934/dirichlet.out

"""
