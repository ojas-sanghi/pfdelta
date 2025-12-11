import matplotlib.pyplot as plt
import seaborn as sns
import torch

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

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    sns.set(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(10, 6))

    # Plot histograms
    for (root, errors), color in zip(per_model.items(), colors):
        pbl_maxs = errors["pbl_maxs"]
        model_name = root_to_model[root]
        sns.histplot(
            pbl_maxs,
            bins=130,
            color=color,
            label=model_name,
            kde=False,
            stat="count",
            alpha=0.6
        )

    plt.xlabel("Power Balance Loss (Max)", fontsize=14, weight="bold")
    plt.xlim(0, 20)
    plt.ylabel("Frequency", fontsize=14, weight="bold")
    plt.title("Frequency of Power Balance Loss (Max) in Test Sets", fontsize=18, weight="bold")
    plt.legend(title="Model", fontsize=12, title_fontsize=13)
    plt.tight_layout()
    plt.savefig("figures/failure_max.svg")
    plt.show()
