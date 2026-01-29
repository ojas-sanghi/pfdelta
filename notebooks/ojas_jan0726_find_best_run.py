import os
import sys
import json
import numpy as np

# Change working directory to one above
sys.path.append(os.getcwd())

cases = ["case14", "case30", "case57", "case118", "case500"]



def parse_path(root):
    parts = root.split("/")
    
    print("PATH PARTS: " + str(parts))

    task = parts[8]
    model = parts[9]
    if not parts[-1].startswith("seed_"):
        return task, model, -1, ""

    run_parts = parts[-1].split("_")

    seed = int(run_parts[1])

    if task in ["task31", "task34"]:
        case = run_parts[2]
    elif task in ["task32", "task33"]:
        case = -1
    
    print(model)
    print(task)
    print(seed)
    print(case)
    
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
            case_index = cases.index(case)
        elif task in ["task32", "task33"]:
            case_index = 0
        elif task == "task34":
            # for a combined set like 30/57/118, calculate average over those three
            # to accomplish this, set up a list of the indices
            # then later on we will recognize that it is a list and will average over those values
            case_numbers = case.split("-")
            case_names = [
                case_numbers[0],
                "case" + case_numbers[1],
                "case" + case_numbers[2],
            ]
            case_index = [cases.index(cn) for cn in case_names]

        if isinstance(case_index, list):
            # average over multiple cases
            this_error = 0.0
            for ci in case_index:
                this_error += val_list[ci][error_key]
            this_error /= len(case_index)
        else:
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
