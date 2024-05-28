import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def to_sequences(data, seq_length, device):
    x = []
    y = []

    for i in range(len(data) - seq_length):

        _x = data[i : (i + seq_length)]
        _y = float(data[i + seq_length])
        x_tensor = torch.FloatTensor(_x)
        y_tensor = torch.FloatTensor(
            [
                _y,
            ]
        )
        x.append(x_tensor)
        y.append(y_tensor)
    x_out = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0)
    x_out = x_out.unsqueeze(-1).squeeze()
    y_out = torch.FloatTensor(y)
    print(x_out.shape)
    print(y_out.shape)
    return x_out.to(device), y_out.to(device)


class LSTMForecaster(nn.Module):

    def __init__(self, model_hyperparameters):

        self.input_dim = model_hyperparameters["input_dim"]
        self.hidden_dim = model_hyperparameters["hidden_dim"]
        self.num_layers = model_hyperparameters["num_layers"]
        self.output_dim = model_hyperparameters["output_dim"]

        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(
            self.input_dim, self.hidden_dim, self.num_layers, batch_first=True
        )
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x, device):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        h0 = h0.to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c0 = c0.to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.to(device)


def train_lstm(criterion, optimizer, model, train_X, train_y, device, num_epochs=100):
    epoch_and_loss_list = [["epoch", "loss"]]
    print(f"num_epochs={num_epochs}")
    for epoch in range(num_epochs):
        outputs = model(train_X.unsqueeze(-1), device).squeeze()
        outputs = outputs.to(device)

        optimizer.zero_grad()
        epoch_loss = criterion(outputs, train_y)
        print(f"epoch={(epoch+1)}, epoch_loss={epoch_loss}")
        epoch_loss.backward()
        optimizer.step()
        epoch_and_loss_list.append([(epoch + 1), float(epoch_loss)])
    return epoch_and_loss_list
