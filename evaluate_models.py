import sys
import os
import argparse
import pandas as pd
import torch

from SuperResolution.models import calculate_test_loss, load_model, check_performance
from SuperResolution.utils import load_config, copy_config_file


def clean_tensor_values(value):
    if isinstance(value, torch.Tensor):
        return value.item()
    return value


metrics = []
for name in os.listdir("models"):
    net, config = load_model(os.path.join("models", name))
    metric = calculate_test_loss(net, config, name)
    for i in [2, 3, 4, 5, 6]:
        # Check performance
        L2Hmodelerror, L2Hinterp = check_performance(
            net=net,
            config=config,
            datafolder=f"/home/thelfer1/scr4_tedwar42/thelfer1/high_end_data_{i}/outputXdata_level{config['res_level']}_step00[04]*.dat",
        )
        factor = L2Hinterp / L2Hmodelerror
        metric[f"factor{i}"] = factor
        metric[f"L2Hinterp{i}"] = L2Hinterp
        metric[f"L2Hmodelerror{i}"] = L2Hmodelerror
    metrics.append(metric)
    df = pd.DataFrame(metrics)
    df = df.applymap(clean_tensor_values)
    df.to_csv("metrics_results.csv", index=False)
