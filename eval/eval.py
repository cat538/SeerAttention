import os
import torch
import argparse
import subprocess
import sys
sys.path.append("..")
from proj_config import *

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    ava_models = list(MODEL_ID2PATH.keys())
    ava_datasets = list(DATA_ID2PATH.keys())

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available subcommands")

    # ppl subcommand
    ppl = subparsers.add_parser("ppl", help="Calculate perplexity")
    ppl.add_argument("--seed", type=int, default=42)
    ppl.add_argument("-m", "--model", type=str, default="qwen2.5-3b", choices=ava_models, help="ID of trained model")
    ppl.add_argument("-d", "--dataset", type=str, default="wiki", choices=ava_datasets, help="Dataset name")
    ppl.add_argument("--length", nargs="+", type=int, default=[4096], help="Sequence lengths to evaluate")
    
    ppl.add_argument("--use_seer", action="store_true", help="Use SeerAttn")
    ppl.add_argument("--sp_method", choices=['threshold', 'nz_ratio'], default='threshold')
    ppl.add_argument("--threshold", type=str, default="0.001")
    ppl.add_argument("--nz_ratios", type=str, default="0.5")
    ppl.add_argument("--gate_type", type=str, default="Qavg_Kmaxminavg")
    
    ppl.add_argument("--batch", type=int, default=1, help="Batch size for evaluation")
    ppl.add_argument("--save_dir", type=str, default=f"{PROJ_ROOT}/eval-out/ppl", help="Path to save results")

    args = parser.parse_args()

    if args.command == "ppl":
        N_GPUS_PER_TASK = 1
        num_device_groups = num_gpus // N_GPUS_PER_TASK
        num_tasks = len(args.length)
        device_groups = (torch.arange(0, N_GPUS_PER_TASK)[None, :] + N_GPUS_PER_TASK * torch.arange(num_device_groups)[:, None]).tolist()
        device_groups = [','.join([str(x) for x in group]) for group in device_groups]
        device_group_to_task = [[] for _ in range(num_device_groups)]
        for i, length in enumerate(args.length):
            device_group_to_task[i % num_device_groups].append(length)
        device_group_to_task = [','.join([str(x) for x in group]) for group in device_group_to_task]

        print(device_groups)
        print(device_group_to_task)

        processes = []

        for group, test_lengths in zip(device_groups, device_group_to_task):
            if len(test_lengths) == 0: continue
            command = [
                f"CUDA_VISIBLE_DEVICES={group}",
                "python3",
                f"{PROJ_ROOT}/eval/ppl/eval_ppl.py",
                f"-m {args.model}",
                f"-d {args.dataset}",
                f"--length {test_lengths}",
                f"--batch {args.batch}",
                f"--save_dir {args.save_dir}",
                f"--seed {args.seed}",
                f"--gate_type {args.gate_type}"
                f"--sparsity_method {args.sp_method}",
                f"--threshold {args.threshold}",
                f"--nz_ratios {args.nz_ratios}",
                f"--use_seer" if args.use_seer else "",
            ]
            command_str = " ".join(command)
            print(f"Running command: {command_str}")
            process = subprocess.Popen(command_str, shell=True)
            processes.append(process)

        for process in processes:
            process.wait()