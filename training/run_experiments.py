#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 mailitg <mailitg@maili-mba.home>
#
# Distributed under terms of the MIT license.

import os, random, subprocess, re

N_RUNS = 10

batch_sizes = [128, 256]
optimizers = ["ADAMW", "SGD"]
activations = ["silu", "relu"]

def sample_lr(opt):
    if opt == "ADAMW":
        return 10 ** random.uniform(-3.5, -2.5)  
    else: 
        return 10 ** random.uniform(-1.7, -0.5)  

def run_model(script, params):
    env = os.environ.copy()
    env.update({k: str(v) for k, v in params.items()})
    print(f"\n>>> Running {script} with {params}")
    result = subprocess.run(["python", script], capture_output=True, text=True, env=env)
    out = result.stdout
    match = re.search(r"Test Accuracy:\s*([\d.]+)%", out)
    acc = float(match.group(1)) if match else None
    print(f"Result: {acc}%\n")
    return acc, out

def log_result(model, params, acc):
    with open("hyperparameters.md", "a") as f:
        f.write(f"| {model} | {params.get('BATCH')} | {params.get('OPT')} | {params.get('LR_MAX'):.4g} | {params.get('ACT')} | {acc:.2f}% |\n")

# run MLP experiments 
for _ in range(N_RUNS):
    params = {
        "BATCH": random.choice(batch_sizes),
        "OPT": random.choice(optimizers),
        "ACT": random.choice(activations),
    }
    params["LR_MAX"] = sample_lr(params["OPT"])
    acc, _ = run_model("mnist_mlp.py", params)
    if acc is not None:
        log_result("MLP", params, acc)

#  run CNN experiments 
for _ in range(N_RUNS):
    params = {
        "BATCH": random.choice(batch_sizes),
        "OPT": random.choice(optimizers),
        "ACT": random.choice(activations),
    }
    params["LR_MAX"] = sample_lr(params["OPT"])
    acc, _ = run_model("mnist_convnet.py", params)
    if acc is not None:
        log_result("CNN", params, acc)


