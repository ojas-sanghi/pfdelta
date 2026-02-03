import os
import sys
import json
import numpy as np

# Change working directory to one above
sys.path.append(os.getcwd())

cases = ["case14", "case30", "case57", "case118", "case500"]
models = ["graphconv", "powerflownet"]


def parse_path(root):
    parts = root.split("/")

    # print("PATH PARTS: " + str(parts))

    task = [t for t in parts if t.startswith("task")][0]
    model = [m for m in parts if m in models][0]

    if not parts[-1].startswith("seed="):
        return task, model, -1, ""

    run_parts = parts[-1].split("_")

    # print("RUN PARTS: " + str(run_parts))

    seed = int(run_parts[0].split("=")[1])

    if task in ["task31", "task34"]:
        case = run_parts[1].split("=")[1]
    elif task in ["task32", "task33"]:
        case = -1

    # print(model)
    # print(task)
    # print(seed)
    # print(case)

    return model, task, seed, case


def find_best_run(root_folder, error_key):
    best_err_per_seed = [float("inf"), float("inf"), float("inf")]
    best_run_path_per_seed = [None, None, None]
    best_summary_per_seed = [None, None, None]
    best_err_mean = -1
    best_err_std = -1

    # Traverse the root directory
    for root, dirs, files in os.walk(root_folder):
        model, task, seed, case = parse_path(root)

        # print("ROOTS: " + root)
        # print("DIRS: " + str(dirs))
        # print("FILES: " + str(files))
        # print(model, task, seed, case)

        # Check for summary.json file in each run folder
        if "summary.json" not in files:
            continue

        seed_index = seed % 42

        summary_path = os.path.join(root, "summary.json")

        try:
            with open(summary_path, "r") as f:
                summary_data = json.load(f)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error reading {summary_path}: {e}")

        if error_key == "_first_one":
            sample_val = summary_data["val"][0]
            val_errors = list(sample_val.keys())
            error_key = val_errors[0]

        val_list = summary_data["val"]

        if task == "task31":
            # case_index depends on current case
            # plus one becauswe we have an extra val set that corresponds to the trian set
            # only after that do we have a fixed order for the cases that we can index to
            case_index = cases.index(case) + 1
        else:
            # for task32 and task33, the val set is the first one
            # and also for task34 because of the connected_list we establish
            case_index = 0

        this_error = val_list[case_index][error_key]

        # If the current run has a lower error, update the best run
        if this_error < best_err_per_seed[seed_index]:
            best_err_per_seed[seed_index] = this_error
            best_run_path_per_seed[seed_index] = root
            best_summary_per_seed[seed_index] = summary_data

        best_err_mean = np.mean(best_err_per_seed)
        best_err_std = np.std(best_err_per_seed)

    return (
        best_run_path_per_seed,
        best_err_per_seed,
        best_err_mean,
        best_err_std,
        best_summary_per_seed,
        error_key,
    )


if __name__ == "__main__":
    root_path = "/mnt/home/donti-group-shared/ojas/pfdelta/runs/gen_feb0126"
    task = "task31"
    model = "graphconv"
    case = "case14"
    error_key = "PBL Mean"

    folder = f"{root_path}/{task}/{model}/{case}"

    best_runs = find_best_run(folder, error_key)

    for r in best_runs:
        print(r)

    # (
    #     best_run_path_per_seed,
    #     best_err_per_seed,
    #     best_err_mean,
    #     best_err_std,
    #     best_summary_per_seed,
    #     error_key,
    # ) = find_best_run(root_folder, error_key)

    # print("Best run paths per seed:", best_run_path_per_seed)
    # print("Best errors per seed:", best_err_per_seed)
    # print("Best error mean:", best_err_mean)
    # print("Best error std:", best_err_std)
