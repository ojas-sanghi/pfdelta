import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

root_to_model = {
    "runs/pfnet_task_1_3": "PFNet",
    "runs/canos_task_1_3": "CANOS-PF",
    "runs/gns_task_1_3": "GNS-S",
}

if __name__ == "__main__":
    per_model = torch.load(
        "values_for_failure_analysis.pkl",
        map_location=torch.device("cpu")
    )

    # consistent style
    sns.set(style="whitegrid", font_scale=1.2)
    colors = {
        "PQs": "#1f77b4",    # blue
        "PVs": "#ff7f0e",    # orange
        "Slacks": "#2ca02c", # green
    }

    # custom bins per bus type
    bin_config = {
        "PQs": 30,
        "PVs": 60,
        "Slacks": 150,
    }

    for root, errors in per_model.items():
        model_name = root_to_model[root]
        plt.figure(figsize=(10, 6))

        # Get per-type arrays
        bus_type_data = {
            "PQs": np.array(errors["pqs_pbl"]),
            "PVs": np.array(errors["pvs_pbl"]),
            "Slacks": np.array(errors["slacks_pbl"]),
        }

        # Plot histograms
        for name, data in bus_type_data.items():
            # Skip slacks for CANOS-PF
            if model_name == "CANOS-PF" and name == "Slacks":
                continue

            bins = bin_config[name]

            sns.histplot(
                data,
                bins=bins,
                color=colors[name],
                label=name,
                kde=False,
                stat="count",
                alpha=0.6
            )

        plt.xlabel("Power Balance Loss", fontsize=14)
        plt.xlim(0, 3.5)
        plt.ylabel("Frequency", fontsize=14)
        plt.title(f"Power Balance Loss per Bus Type in {model_name}", fontsize=16, weight="bold")
        plt.legend(title="Bus Type", fontsize=12, title_fontsize=13)
        plt.tight_layout()
        plt.savefig(f"figures/per_bus_type_{model_name}.svg")
        plt.show()
