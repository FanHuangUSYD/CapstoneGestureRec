import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from LSTM.DataGenerator.get_ecg_data import get_data_loader
from LSTM.util import get_directory, device, plot_accuracy, plot_confusion_matrix


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
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


def train_model(model, train_loader, val_loader, test_loader, name_list: list[str], num_epochs: int = 200, learning_rate: float = 0.001,
                early_stop_epochs: int = 30, train_name: str = "test"):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_accuracy = 0.0
    no_improvement_count = 0

    train_accuracies = []  # List to store training accuracies
    val_accuracies = []  # List to store validation accuracies

    save_path = get_directory()

    best_model = os.path.join(save_path, train_name + '_best.pt')
    last_model = os.path.join(save_path, train_name + '_last.pt')

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
        train_accuracies.append(accuracy)
        val_accuracies.append(val_accuracy)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            no_improvement_count = 0
            torch.save(model, best_model)  # Save the best model
        else:
            no_improvement_count += 1

        if no_improvement_count >= early_stop_epochs:
            print(f'Early stopping at epoch {epoch + 1} due to no improvement on validation accuracy. Best Accuracy: {best_accuracy:.4f}')
            break
        if 1 - accuracy < 10e-10 and 1 - val_accuracy < 10e-10:
            print(f'Early stopping at epoch {epoch + 1} due to both training and validation accuracy have reached their highest values. '
                  f'Best Accuracy: {best_accuracy:.4f}')
            break

    torch.save(model, last_model)  # Save the final model
    plot_accuracy({"Training Dataset": train_accuracies, "Validation Dataset": val_accuracies})
    test_model(best_model, output_size, test_loader, name_list)
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


def test_model(model_path: str, output_size: int, dataloader, name_list: list[str]) -> None:

    target_model = torch.load(model_path)
    target_model.eval()

    confusion_matrix = np.zeros((output_size, output_size), dtype=float)

    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = target_model(inputs)
            _, predicted = torch.max(outputs, 1)

            for i in range(len(targets)):
                confusion_matrix[predicted[i], targets[i]] += 1

    # column sum
    col_sum = confusion_matrix.sum(axis=0)
    # Iterate through each element and divide by the sum of its column
    for i in range(output_size):
        confusion_matrix[:, i] /= col_sum[i]

    # Plot the confusion matrix
    plot_confusion_matrix(confusion_matrix, name_list)
    return


if __name__ == '__main__':
    # initialize model
    input_size = 2
    hidden_size = 64
    output_size = 5

    # initialize dataloader
    seq_length = 4000
    batch_size = 10

    # labels
    name_list = ["ECG Data"] + ["Mock Data " + str(i) for i in range(1, output_size)]

    # train model
    # num_epochs = 100
    # learning_rate = 0.001

    model = LSTMModel(input_size, hidden_size, output_size)

    train_loader = get_data_loader(name_list, num_samples=6000, feature_num=input_size, seq_length=seq_length,
                                   batch_size=batch_size, data_class=output_size)

    val_loader = get_data_loader(name_list, num_samples=2000, feature_num=input_size, seq_length=seq_length,
                                 batch_size=batch_size, data_class=output_size, tag="Validation Dataset")

    test_loader = get_data_loader(name_list, num_samples=1000, feature_num=input_size, seq_length=seq_length,
                                  batch_size=batch_size, data_class=output_size, tag="Test Dataset")

    train_model(model, train_loader, val_loader, test_loader, name_list, train_name="LSTM_Validation")

