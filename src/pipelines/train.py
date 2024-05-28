from functools import partial
import json
from os import path
import torch

from torch.utils.data import DataLoader
from ruamel.yaml import YAML
import pendulum

import pandas as pd
from model.model import get_device, LSTMForecaster, to_sequences, train_lstm

yaml = YAML()


def _get_project_dir_folder():
    return path.dirname((path.dirname(path.dirname(__file__))))


def _construct_and_save_report(config, losses, now, later, model, model_json_fp):
    report = {}
    report["name"] = config["name"]
    with open(model_json_fp, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

    return report


#     report["expriment"] = config["experiment"]
#     report = report | config["params"]["common"]
#     report = report | config["params"]["train"]

#     report["pipeline_time_start"] = now.to_datetime_string()
#     report["pipeline_time_end"] = later.to_datetime_string()
#     report["training_time_taken_minutes"] = (later - now).in_minutes()
#     report["training_loss_curve"] = str(losses)
#     report["model_architecture"] = str(model.eval())
#     report["model_count_parameters"] = count_parameters(model)
#     report["model_vocab_size"] = vocab_size
#     report["model_word_window_size"] = word_window_size * 2 + 1
#     return report


def main():

    now = pendulum.now()
    model_timestamp = now.format("YYYYMMDDTHHmmss")
    torch.manual_seed(42)
    device = get_device()

    experiment = "baseline"
    ASSETS_FP = path.join(_get_project_dir_folder(), "assets")

    signal_fp = path.join(
        ASSETS_FP,
        "datasets",
        f"x.csv",
    )
    config_fp = path.join(ASSETS_FP, "config", f"config-{experiment}.yaml")

    model_fp = path.join(
        ASSETS_FP,
        "models",
        f"model-{experiment}-{model_timestamp}.pkl",
    )
    model_json_fp = path.join(
        ASSETS_FP,
        "models",
        f"modelprofile-{experiment}-{model_timestamp}.json",
    )

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

    train_split = config_train.get("train_split")
    train_idx = int(data.shape[0] * train_split)
    train = data[:train_idx]
    test = data[train_idx:]
    print(train.shape)
    print(test.shape)

    train_X, train_y = to_sequences(train, config_common["sequence_length"], device)
    # Model
    model_hyperparameters = {
        "input_dim": config_common["input_dim"],
        "hidden_dim": config_common["hidden_dim"],
        "output_dim": config_common["output_dim"],
        "num_layers": config_common["num_layers"],
    }
    model = LSTMForecaster(
        model_hyperparameters,
    )
    model = model.to(device)

    # Train model
    learning_rate = config_train["learning_rate"]
    criterion = torch.nn.MSELoss()  # mean-squared error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = config_train["epochs"]
    # Train
    losses = train_lstm(
        criterion, optimizer, model, train_X, train_y, device, num_epochs=num_epochs
    )
    later = pendulum.now()

    # Save trained model & environment
    torch.save(model.state_dict(), model_fp)
    j = _construct_and_save_report(
        config,
        losses,
        now,
        later,
        model,
        model_json_fp,
    )

    print(f"Done. Timestamp = {model_timestamp}")


if __name__ == "__main__":
    main()
