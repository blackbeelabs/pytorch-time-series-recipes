from functools import partial
import json
from os import path
import torch

from torch.utils.data import DataLoader
from ruamel.yaml import YAML
import pendulum

import pandas as pd
from model.model import get_device, custom_collate_fn

yaml = YAML()


def _get_project_dir_folder():
    return path.dirname(path.dirname((path.dirname(path.dirname(__file__)))))


def main():

    experiment = "baseline"
    ASSETS_FP = path.join(_get_project_dir_folder(), "assets")
    device = get_device()

    signal_fp = path.join(
        ASSETS_FP,
        "datasets",
        f"x.csv",
    )
    config_fp = path.join(ASSETS_FP, "config", f"config-{experiment}.yaml")

    # Signal
    df = pd.read_csv(signal_fp, sep="\t")
    signal_series = df["x"]
    data = signal_series.values
    print(signal_series.shape)
    # Config
    config = None
    with open(config_fp) as f:
        config = yaml.load(f)
    config_common = config.get("params").get("common")
    config_train = config.get("params").get("train")
    print(config_common)
    print(config_train)

    collate_fn = partial(
        custom_collate_fn, seq_length=config_common["sequence_length"], device=device
    )
    loader = DataLoader(
        data,
        batch_size=len(data),
        collate_fn=collate_fn,
    )

    for xB, yB in loader:
        print(xB.shape)
        print(yB.shape)
        print()


if __name__ == "__main__":
    main()
