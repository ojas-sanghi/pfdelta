import copy
import os
import statistics
import sys
from typing import List

sys.path.append(os.getcwd())
from core.trainers.gnn_trainer import GNNTrainer
from core.utils.registry import registry
from scripts.utils import load_config, load_trainer


def choose_task_number(test_set, my_config):
    selected_task = ""
    if test_set == "case14_30_57_seeds":
        selected_task = 3.2
    elif test_set == "case118_500_seeds":
        selected_task = 3.3
    elif test_set == (my_config + "_seeds"):
        selected_task = 3.1
    else:
        selected_task = 1.3
    return selected_task


graphconv_config_options = {
    "case14_30_57": [
        "gen_msrp_fe/run1/task3.2/graphconv/sweep_layers_15_hidden_256_lr_5e-3_epochs_100_11_250918_172434",
        "gen_msrp_fe/run2/task3.2/graphconv/sweep_layers_15_hidden_256_lr_5e-3_epochs_100_11_250923_141218",
        "gen_msrp_fe/run3/task3.2/graphconv/sweep_layers_15_hidden_256_lr_5e-3_epochs_100_11_250924_053553",
    ],
    "case118_500": [
        "gen_msrp_fe/run1/task33/graphconv/sweep_layers_10_hidden_256_lr_1e-3_epochs_100_6_250918_183035",
        "gen_msrp_fe/run2/task33/graphconv/sweep_layers_10_hidden_256_lr_1e-3_epochs_100_6_250923_151547",
        "gen_msrp_fe/run3/task33/graphconv/sweep_layers_10_hidden_256_lr_1e-3_epochs_100_6_250924_060819",
    ],
    "case14": [
        "gen_msrp_fe/run1/task3.1/case14/graphconv/sweep_case_14_layers_15_hidden_256_lr_1e-3_epochs_100_10_250918_002521",
        "gen_msrp_fe/run2/task3.1/case14/graphconv/sweep_case_14_layers_15_hidden_256_lr_1e-3_epochs_100_10_250923_030825",
        "gen_msrp_fe/run3/task3.1/case14/graphconv/sweep_case_14_layers_15_hidden_256_lr_1e-3_epochs_100_10_250923_225901",
    ],
    "case30": [
        "gen_msrp_fe/run1/task3.1/case30/graphconv/sweep_case_30_layers_10_hidden_256_lr_5e-3_epochs_100_7_250918_134947",
        "gen_msrp_fe/run2/task3.1/case30/graphconv/sweep_case_30_layers_10_hidden_256_lr_5e-3_epochs_100_7_250923_051516",
        "gen_msrp_fe/run3/task3.1/case30/graphconv/sweep_case_30_layers_10_hidden_256_lr_5e-3_epochs_100_7_250923_235814",
    ],
    "case57": [
        "gen_msrp_fe/run1/task3.1/case57/graphconv/sweep_case_57_layers_15_hidden_256_lr_1e-3_epochs_100_10_250918_045804",
        "gen_msrp_fe/run2/task3.1/case57/graphconv/sweep_case_57_layers_15_hidden_256_lr_1e-3_epochs_100_10_250923_080851",
        "gen_msrp_fe/run3/task3.1/case57/graphconv/sweep_case_57_layers_15_hidden_256_lr_1e-3_epochs_100_10_250924_011934",
    ],
    "case118": [
        "gen_msrp_fe/run1/task3.1/case118/graphconv/sweep_case_118_layers_10_hidden_256_lr_5e-3_epochs_100_7_250918_063129",
        "gen_msrp_fe/run2/task3.1/case118/graphconv/sweep_case_118_layers_10_hidden_256_lr_5e-3_epochs_100_7_250923_094040",
        "gen_msrp_fe/run3/task3.1/case118/graphconv/sweep_case_118_layers_10_hidden_256_lr_5e-3_epochs_100_7_250924_021224",
    ],
    "case500": [
        "gen_msrp_fe/run1/task3.1/case500/graphconv/sweep_case_500_layers_15_hidden_256_lr_5e-3_epochs_100_11_250918_125220",
        "gen_msrp_fe/run2/task3.1/case500/graphconv/sweep_case_500_layers_15_hidden_256_lr_5e-3_epochs_100_11_250923_124257",
        "gen_msrp_fe/run3/task3.1/case500/graphconv/sweep_case_500_layers_15_hidden_256_lr_5e-3_epochs_100_11_250924_035421",
    ],
}


powerflownet_config_options = {
    "case14_30_57": [
        "gen_msrp_fe/run1/task3.2/powerflownet/sweep_hidden_256_lr_1e-3_epochs_100_2_250918_204717",
        "gen_msrp_fe/run2/task3.2/powerflownet/sweep_hidden_256_lr_1e-3_epochs_100_2_250923_200035",
        "gen_msrp_fe/run3/task3.2/powerflownet/sweep_hidden_256_lr_1e-3_epochs_100_2_250924_144734",
    ],
    "case118_500": [
        "gen_msrp_fe/run1/task33/powerflownet/sweep_hidden_256_lr_5e-4_epochs_100_1_250918_205439",
        "gen_msrp_fe/run2/task33/powerflownet/sweep_hidden_256_lr_5e-4_epochs_100_1_250923_203328",
        "gen_msrp_fe/run3/task33/powerflownet/sweep_hidden_256_lr_5e-4_epochs_100_1_250924_145748",
    ],
    "case14": [
        "gen_msrp_fe/run1/task3.1/case14/powerflownet/sweep_case_14_hidden_256_lr_5e-3_epochs_100_3_250918_191427",
        "gen_msrp_fe/run2/task3.1/case14/powerflownet/sweep_case_14_hidden_256_lr_5e-3_epochs_100_3_250923_170703",
        "gen_msrp_fe/run3/task3.1/case14/powerflownet/sweep_case_14_hidden_256_lr_5e-3_epochs_100_3_250924_123742",
    ],
    "case30": [
        "gen_msrp_fe/run1/task3.1/case30/powerflownet/sweep_case_30_hidden_256_lr_5e-3_epochs_100_3_250918_194118",
        "gen_msrp_fe/run2/task3.1/case30/powerflownet/sweep_case_30_hidden_256_lr_5e-3_epochs_100_3_250923_174056",
        "gen_msrp_fe/run3/task3.1/case30/powerflownet/sweep_case_30_hidden_256_lr_5e-3_epochs_100_3_250924_123756",
    ],
    "case57": [
        "gen_msrp_fe/run1/task3.1/case57/powerflownet/sweep_case_57_hidden_256_lr_1e-3_epochs_100_2_250918_194457",
        "gen_msrp_fe/run2/task3.1/case57/powerflownet/sweep_case_57_hidden_256_lr_1e-3_epochs_100_2_250923_180506",
        "gen_msrp_fe/run3/task3.1/case57/powerflownet/sweep_case_57_hidden_256_lr_1e-3_epochs_100_2_250924_132935",
    ],
    "case118": [
        "gen_msrp_fe/run1/task3.1/case118/powerflownet/sweep_case_118_hidden_256_lr_1e-4_epochs_100_0_250918_195038",
        "gen_msrp_fe/run2/task3.1/case118/powerflownet/sweep_case_118_hidden_256_lr_1e-4_epochs_100_0_250923_181053",
        "gen_msrp_fe/run3/task3.1/case118/powerflownet/sweep_case_118_hidden_256_lr_1e-4_epochs_100_0_250924_133313",
    ],
    "case500": [
        "gen_msrp_fe/run1/task3.1/case500/powerflownet/sweep_case_500_hidden_256_lr_1e-4_epochs_100_0_250918_201221",
        "gen_msrp_fe/run2/task3.1/case500/powerflownet/sweep_case_500_hidden_256_lr_1e-4_epochs_100_0_250923_183714",
        "gen_msrp_fe/run3/task3.1/case500/powerflownet/sweep_case_500_hidden_256_lr_1e-4_epochs_100_0_250924_134323",
    ],
}


dataset_configs = {
    0: ["case14_seeds", "case30_seeds", "case57_seeds"],
    1: ["case118_seeds", "case500_seeds"],
    2: ["case14_30_57_seeds", "case118_500_seeds"],
}


def do_stuff(config_options, my_config, test_datasets):
    roots = [os.path.join("runs", c) for c in config_options[my_config]]
    seed_configs = [load_config(root) for root in roots]
    trainers: List[GNNTrainer] = [load_trainer(cfg) for cfg in seed_configs]

    new_test_datasets = [
        {
            "add_bus_type": "false",
            "case_name": f"{test_set}--{my_config}_seeds",
            "model": "PFNet",
            "name": "pfdeltaPFNet",
            "root_dir": "/mnt/home/donti-group-shared/hf_pfdelta_data",
            "split": "val",
            "task": choose_task_number(test_set, my_config),
            "transform": "pfnet_data_mean0_var1",
        }
        for test_set in test_datasets
    ]

    print("Loading new datasets...")
    new_combined_datasets = []
    for d in new_test_datasets:
        dataset_name = d["name"]
        dataset_class = registry.get_dataset_class(dataset_name)
        dataset_inputs = copy.deepcopy(d)
        del dataset_inputs["name"]
        dataset = dataset_class(**dataset_inputs)
        new_combined_datasets.append(dataset)
    print("Loaded new datasets!")

    print("Setting up new dataloaders...")
    for trainer in trainers:
        dataloader_class = trainer.get_dataloader_class()
        new_val_params = trainer.config["optim"]["val_params"]
        batch_size = new_val_params["batch_size"]
        for dataset in new_combined_datasets:
            new_dataloader = dataloader_class(
                dataset, batch_size=batch_size, shuffle=False
            )
            trainer.dataloaders.append(new_dataloader)
    print("Set up new dataloaders!")

    results_by_trainer = {}
    for trainer in trainers:
        num_new_dataloaders = -1 * (len(test_datasets))
        combined_dataloaders = trainer.dataloaders[num_new_dataloaders:]

        seeds_running_losses = {}
        for i, combined_d in enumerate(combined_dataloaders):
            seeds_running_losses[combined_d] = []
            seeds_running_losses[combined_d].extend(
                trainer.calc_one_val_error(combined_d, i, max_val=len(test_datasets))
            )

        for dl, running_loss in seeds_running_losses.items():
            for i in range(len(running_loss)):
                running_loss[i] = running_loss[i] / len(dl)

        results_by_trainer[trainer] = seeds_running_losses

    losses_to_analyze = ["PBL Mean", "PBL Max"]

    case_agg = {
        case: {loss: [] for loss in losses_to_analyze} for case in test_datasets
    }

    for tnum, trainer in enumerate(trainers):
        seeds_running_losses = results_by_trainer[trainer]
        dataloader_names = list(seeds_running_losses.keys())
        for i, case in enumerate(test_datasets):
            dataloader_name = dataloader_names[i]
            for loss_name in losses_to_analyze:
                loss_index = trainer.val_loss_names.index(loss_name)
                val = seeds_running_losses[dataloader_name][loss_index]
                case_agg[case][loss_name].append(val)

    print("TRAINED ON: ", my_config)
    for case in test_datasets:
        print(f"Results for {case}:")
        for loss_name in losses_to_analyze:
            print(f"\tLoss: {loss_name}")
            vals = case_agg[case][loss_name]
            print(f"\t\tmean: {statistics.mean(vals)}")
            print(f"\t\tstddev: {statistics.stdev(vals)}")
            print(f"\t\tpoints: {vals}")
        print("---")


print("1-1======================")
my_config = "case14"
dataset_config = dataset_configs[0]
do_stuff(powerflownet_config_options, my_config, dataset_config)


print("1-2======================")
my_config = "case30"
dataset_config = dataset_configs[0]
do_stuff(powerflownet_config_options, my_config, dataset_config)

print("1-3======================")
my_config = "case57"
dataset_config = dataset_configs[0]
do_stuff(powerflownet_config_options, my_config, dataset_config)

print("1-4======================")
my_config = "case14_30_57"
dataset_config = dataset_configs[0]
do_stuff(powerflownet_config_options, my_config, dataset_config)


print("**********************")
print("**********************")
print("**********************")

print("2-1======================")
my_config = "case118"
dataset_config = dataset_configs[1]
do_stuff(powerflownet_config_options, my_config, dataset_config)

print("2-2======================")
my_config = "case500"
dataset_config = dataset_configs[1]
do_stuff(powerflownet_config_options, my_config, dataset_config)

print("2-3======================")
my_config = "case118_500"
dataset_config = dataset_configs[1]
do_stuff(powerflownet_config_options, my_config, dataset_config)


print("**********************")
print("**********************")
print("**********************")

print("3-1======================")
my_config = "case14"
dataset_config = dataset_configs[2]
do_stuff(powerflownet_config_options, my_config, dataset_config)

print("3-2======================")
my_config = "case30"
dataset_config = dataset_configs[2]
do_stuff(powerflownet_config_options, my_config, dataset_config)

print("3-3======================")
my_config = "case57"
dataset_config = dataset_configs[2]
do_stuff(powerflownet_config_options, my_config, dataset_config)

print("3-4======================")
my_config = "case118"
dataset_config = dataset_configs[2]
do_stuff(powerflownet_config_options, my_config, dataset_config)

print("3-5======================")
my_config = "case500"
dataset_config = dataset_configs[2]
do_stuff(powerflownet_config_options, my_config, dataset_config)

print("3-6======================")
my_config = "case14_30_57"
dataset_config = dataset_configs[2]
do_stuff(powerflownet_config_options, my_config, dataset_config)

print("3-7======================")
my_config = "case118_500"
dataset_config = dataset_configs[2]
do_stuff(powerflownet_config_options, my_config, dataset_config)
