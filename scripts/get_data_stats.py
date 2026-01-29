from core.datasets.pfdelta_variants import PFDeltaPFNet as dataset


def get_data_for_case(case_name, task_num):
    ds = dataset(
        root_dir="/mnt/home/donti-group-shared/pfdelta_data",
        task=task_num,
        case_name=case_name,
    )

    return_dict = {
        case_name: {
            "casename": case_name,
            "mean": {
                "bus": {"y": ds._data["bus"].y.mean(dim=0)},
                ("bus", "branch", "bus"): {
                    "edge_attr": ds._data[("bus", "branch", "bus")].edge_attr.mean(
                        dim=0
                    )
                },
            },
            "std": {
                "bus": {"y": ds._data["bus"].y.std(dim=0)},
                ("bus", "branch", "bus"): {
                    "edge_attr": ds._data[("bus", "branch", "bus")].edge_attr.std(dim=0)
                },
            },
        }
    }

    return return_dict


# uv run python -m scripts.get_data_stats
if __name__ == "__main__":
    print("Loading dataset...")

    this_casename = "case30_57_500"
    task_num = 3.4

    da = get_data_for_case(this_casename, task_num)
    da_str = str(da)
    da_str = da_str.replace("tensor", "torch.tensor")

    print(da_str)
