#!/usr/bin/env python3
"""
Monitor running experiments and automatically start the next one in the queue.
"""

import subprocess
import time
import json
import os
from pathlib import Path

# Define the queue of experiments to run
EXPERIMENT_QUEUE = [
    # Experiment 1.1: K=32 or K=64 fixed, L=2,4,8,16 varying (8 runs total)
    ["python", "-m", "hw2.experiments", "run-exp", "-n", "exp1_1_L2_K32", "-K", "32", "-L", "2", "-P", "2", "-H", "100", "--batches", "100", "--epochs", "100", "--early-stopping", "3"],
    ["python", "-m", "hw2.experiments", "run-exp", "-n", "exp1_1_L4_K32", "-K", "32", "-L", "4", "-P", "2", "-H", "100", "--batches", "100", "--epochs", "100", "--early-stopping", "3"],
    ["python", "-m", "hw2.experiments", "run-exp", "-n", "exp1_1_L8_K32", "-K", "32", "-L", "8", "-P", "4", "-H", "100", "--batches", "100", "--epochs", "100", "--early-stopping", "3"],
    ["python", "-m", "hw2.experiments", "run-exp", "-n", "exp1_1_L16_K32", "-K", "32", "-L", "16", "-P", "8", "-H", "100", "--batches", "100", "--epochs", "100", "--early-stopping", "3"],
    ["python", "-m", "hw2.experiments", "run-exp", "-n", "exp1_1_L2_K64", "-K", "64", "-L", "2", "-P", "2", "-H", "100", "--batches", "100", "--epochs", "100", "--early-stopping", "3"],
    ["python", "-m", "hw2.experiments", "run-exp", "-n", "exp1_1_L4_K64", "-K", "64", "-L", "4", "-P", "2", "-H", "100", "--batches", "100", "--epochs", "100", "--early-stopping", "3"],
    ["python", "-m", "hw2.experiments", "run-exp", "-n", "exp1_1_L8_K64", "-K", "64", "-L", "8", "-P", "4", "-H", "100", "--batches", "100", "--epochs", "100", "--early-stopping", "3"],
    ["python", "-m", "hw2.experiments", "run-exp", "-n", "exp1_1_L16_K64", "-K", "64", "-L", "16", "-P", "8", "-H", "100", "--batches", "100", "--epochs", "100", "--early-stopping", "3"],

    # Experiment 1.2: L=2,4,8 fixed, K=[32],[64],[128] varying (9 runs total)
    ["python", "-m", "hw2.experiments", "run-exp", "-n", "exp1_2_L2_K32", "-K", "32", "-L", "2", "-P", "2", "-H", "100", "--batches", "100", "--epochs", "100", "--early-stopping", "3"],
    ["python", "-m", "hw2.experiments", "run-exp", "-n", "exp1_2_L2_K64", "-K", "64", "-L", "2", "-P", "2", "-H", "100", "--batches", "100", "--epochs", "100", "--early-stopping", "3"],
    ["python", "-m", "hw2.experiments", "run-exp", "-n", "exp1_2_L2_K128", "-K", "128", "-L", "2", "-P", "2", "-H", "100", "--batches", "100", "--epochs", "100", "--early-stopping", "3"],
    ["python", "-m", "hw2.experiments", "run-exp", "-n", "exp1_2_L4_K32", "-K", "32", "-L", "4", "-P", "2", "-H", "100", "--batches", "100", "--epochs", "100", "--early-stopping", "3"],
    ["python", "-m", "hw2.experiments", "run-exp", "-n", "exp1_2_L4_K64", "-K", "64", "-L", "4", "-P", "2", "-H", "100", "--batches", "100", "--epochs", "100", "--early-stopping", "3"],
    ["python", "-m", "hw2.experiments", "run-exp", "-n", "exp1_2_L4_K128", "-K", "128", "-L", "4", "-P", "2", "-H", "100", "--batches", "100", "--epochs", "100", "--early-stopping", "3"],
    ["python", "-m", "hw2.experiments", "run-exp", "-n", "exp1_2_L8_K32", "-K", "32", "-L", "8", "-P", "4", "-H", "100", "--batches", "100", "--epochs", "100", "--early-stopping", "3"],
    ["python", "-m", "hw2.experiments", "run-exp", "-n", "exp1_2_L8_K64", "-K", "64", "-L", "8", "-P", "4", "-H", "100", "--batches", "100", "--epochs", "100", "--early-stopping", "3"],
    ["python", "-m", "hw2.experiments", "run-exp", "-n", "exp1_2_L8_K128", "-K", "128", "-L", "8", "-P", "4", "-H", "100", "--batches", "100", "--epochs", "100", "--early-stopping", "3"],

    # Experiment 1.3: K=[64,128] fixed, L=2,3,4 varying (3 runs total)
    ["python", "-m", "hw2.experiments", "run-exp", "-n", "exp1_3_L2_K64-128", "-K", "64", "128", "-L", "2", "-P", "2", "-H", "100", "--batches", "100", "--epochs", "100", "--early-stopping", "3"],
    ["python", "-m", "hw2.experiments", "run-exp", "-n", "exp1_3_L3_K64-128", "-K", "64", "128", "-L", "3", "-P", "3", "-H", "100", "--batches", "100", "--epochs", "100", "--early-stopping", "3"],
    ["python", "-m", "hw2.experiments", "run-exp", "-n", "exp1_3_L4_K64-128", "-K", "64", "128", "-L", "4", "-P", "4", "-H", "100", "--batches", "100", "--epochs", "100", "--early-stopping", "3"],

    # Experiment 1.4: ResNet - K=[32] L=8,16,32 AND K=[64,128,256] L=2,4,8 (6 runs total)
    ["python", "-m", "hw2.experiments", "run-exp", "-n", "exp1_4_L8_K32", "--model-type", "resnet", "-K", "32", "-L", "8", "-P", "4", "-H", "100", "--batches", "100", "--epochs", "100", "--early-stopping", "3"],
    ["python", "-m", "hw2.experiments", "run-exp", "-n", "exp1_4_L16_K32", "--model-type", "resnet", "-K", "32", "-L", "16", "-P", "8", "-H", "100", "--batches", "100", "--epochs", "100", "--early-stopping", "3"],
    ["python", "-m", "hw2.experiments", "run-exp", "-n", "exp1_4_L32_K32", "--model-type", "resnet", "-K", "32", "-L", "32", "-P", "16", "-H", "100", "--batches", "100", "--epochs", "100", "--early-stopping", "3"],
    ["python", "-m", "hw2.experiments", "run-exp", "-n", "exp1_4_L2_K64-128-256", "--model-type", "resnet", "-K", "64", "128", "256", "-L", "2", "-P", "2", "-H", "100", "--batches", "100", "--epochs", "100", "--early-stopping", "3"],
    ["python", "-m", "hw2.experiments", "run-exp", "-n", "exp1_4_L4_K64-128-256", "--model-type", "resnet", "-K", "64", "128", "256", "-L", "4", "-P", "4", "-H", "100", "--batches", "100", "--epochs", "100", "--early-stopping", "3"],
    ["python", "-m", "hw2.experiments", "run-exp", "-n", "exp1_4_L8_K64-128-256", "--model-type", "resnet", "-K", "64", "128", "256", "-L", "8", "-P", "8", "-H", "100", "--batches", "100", "--epochs", "100", "--early-stopping", "3"],

    # Experiment 2: YourCNN - K=[32,64,128] L=3,6,9,12 (4 runs total)
    ["python", "-m", "hw2.experiments", "run-exp", "-n", "exp2_L3_K32-64-128", "--model-type", "yourcnn", "-K", "32", "64", "128", "-L", "3", "-P", "3", "-H", "100", "--batches", "100", "--epochs", "100", "--early-stopping", "3"],
    ["python", "-m", "hw2.experiments", "run-exp", "-n", "exp2_L6_K32-64-128", "--model-type", "yourcnn", "-K", "32", "64", "128", "-L", "6", "-P", "6", "-H", "100", "--batches", "100", "--epochs", "100", "--early-stopping", "3"],
    ["python", "-m", "hw2.experiments", "run-exp", "-n", "exp2_L9_K32-64-128", "--model-type", "yourcnn", "-K", "32", "64", "128", "-L", "9", "-P", "9", "-H", "100", "--batches", "100", "--epochs", "100", "--early-stopping", "3"],
    ["python", "-m", "hw2.experiments", "run-exp", "-n", "exp2_L12_K32-64-128", "--model-type", "yourcnn", "-K", "32", "64", "128", "-L", "12", "-P", "12", "-H", "100", "--batches", "100", "--epochs", "100", "--early-stopping", "3"],
]

def get_gpu_processes():
    """Get list of PIDs using the GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        pids = [int(line.strip()) for line in result.stdout.strip().split('\n') if line.strip()]
        return pids
    except:
        return []

def is_experiment_running():
    """Check if any Python experiment is currently running on the GPU."""
    pids = get_gpu_processes()
    return len(pids) > 0

def experiment_exists(exp_name):
    """Check if experiment results already exist."""
    results_dir = Path("./results")
    if not results_dir.exists():
        return False

    # Check for any JSON file containing the experiment name
    for json_file in results_dir.glob(f"{exp_name}*.json"):
        if json_file.is_file() and json_file.stat().st_size > 0:
            return True
    return False

def run_experiment(cmd):
    """Run a single experiment and wait for it to complete."""
    exp_name = cmd[cmd.index("-n") + 1] if "-n" in cmd else "unknown"
    print(f"\n{'='*80}")
    print(f"Starting experiment: {exp_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    # Run the experiment
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Stream output
    for line in process.stdout:
        print(line, end='')

    process.wait()

    print(f"\n{'='*80}")
    print(f"Experiment {exp_name} completed with exit code: {process.returncode}")
    print(f"{'='*80}\n")

    return process.returncode

def main():
    print("Experiment Monitor and Queue Manager")
    print("=" * 80)
    print(f"Total experiments in queue: {len(EXPERIMENT_QUEUE)}")
    print("=" * 80)

    # Check if an experiment is already running
    if is_experiment_running():
        print("\n[WARNING] Detected experiment already running on GPU!")
        print("Waiting for it to complete before starting queue...\n")

        # Wait for current experiment to finish
        while is_experiment_running():
            time.sleep(30)  # Check every 30 seconds
            print(".", end='', flush=True)

        print("\n[DONE] Current experiment completed!\n")

    # Run each experiment in the queue
    completed_count = 0
    skipped_count = 0

    for i, cmd in enumerate(EXPERIMENT_QUEUE, 1):
        exp_name = cmd[cmd.index("-n") + 1] if "-n" in cmd else f"experiment_{i}"

        print(f"\n[{i}/{len(EXPERIMENT_QUEUE)}] Preparing to run: {exp_name}")

        # Check if experiment already exists
        if experiment_exists(exp_name):
            print(f"[SKIP] Results already exist for {exp_name}")
            skipped_count += 1
            continue

        # Wait a bit between experiments to ensure GPU memory is freed
        time.sleep(5)

        # Run the experiment
        exit_code = run_experiment(cmd)

        if exit_code != 0:
            print(f"[WARNING] Experiment {exp_name} failed with exit code {exit_code}")
            print("Continue with next experiment? (y/n): ", end='', flush=True)
            # In automated mode, we'll continue anyway
            # response = input().strip().lower()
            # if response != 'y':
            #     print("Stopping queue execution.")
            #     break

        completed_count += 1
        print(f"[DONE] Completed {completed_count} new experiments (skipped {skipped_count})")
        print(f"Remaining: {len(EXPERIMENT_QUEUE) - i}")

    print("\n" + "=" * 80)
    print("All experiments completed!")
    print(f"New experiments run: {completed_count}")
    print(f"Experiments skipped (already completed): {skipped_count}")
    print("=" * 80)

if __name__ == "__main__":
    main()
