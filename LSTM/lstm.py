import os

import torch
import torch.nn as nn
import torch.optim as optim

from LSTM.DataGenerator.get_ecg_data import get_data_loader
from LSTM.util import get_directory, device


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # initialise
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)

        # forward
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take the output of the last time point in the sequence as the prediction

        return out


def train_model(model, train_loader, val_loader, num_epochs: int = 100, learning_rate: float = 0.001,
                early_stop_epochs: int = 20, train_name: str = "test"):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_accuracy = 0.0
    no_improvement_count = 0

    save_path = get_directory()

    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        accuracy = correct / total

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        val_accuracy = evaluate_model(model, val_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Accuracy: {accuracy:.4f} | Validation Accuracy: {val_accuracy:.4f}')

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            no_improvement_count = 0
            torch.save(model, os.path.join(save_path, train_name + '_best.pt'))  # Save the best model
        else:
            no_improvement_count += 1

        if no_improvement_count >= early_stop_epochs:
            print(f'Early stopping at epoch {epoch + 1} due to no improvement on validation accuracy. Best Accuracy: {best_accuracy:.4f}')
            break
        if 1 - accuracy < 10e-10 and 1 - val_accuracy < 10e-10:
            print(f'Early stopping at epoch {epoch + 1} due to both training and validation accuracy have reached their highest values. '
                  f'Best Accuracy: {best_accuracy:.4f}')
            break

    torch.save(model, os.path.join(save_path, train_name + '_last.pt'))  # Save the final model
    return


def evaluate_model(model, dataloader) -> float:
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = correct / total
    return accuracy


if __name__ == '__main__':
    # initialize model
    input_size = 2
    hidden_size = 64
    output_size = 5

    # initialize dataloader
    seq_length = 4000
    batch_size = 10

    # train model
    # num_epochs = 100
    # learning_rate = 0.001

    model = LSTMModel(input_size, hidden_size, output_size)

    train_loader = get_data_loader(num_samples=6000, feature_num=input_size, seq_length=seq_length, batch_size=batch_size, data_class=output_size)
    val_loader = get_data_loader(num_samples=1000, feature_num=input_size, seq_length=seq_length, batch_size=batch_size, data_class=output_size)

    train_model(model, train_loader, val_loader)

# 实验；
# train dta 周期  normal distribu(bias)
# 一段一段测，（长的截取）设定长度的大小（希望小的也性能好）段的大小，

# 训练数据用不同长度 （dimension 一样 ）
