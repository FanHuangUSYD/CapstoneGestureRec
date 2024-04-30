import os
from collections import Counter
from datetime import datetime
from typing import Dict, List

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch import device

ROOT = os.path.dirname(__file__)


def get_directory() -> str:
    now = datetime.now()
    month_day = now.strftime("%m-%d")

    directory_path = os.path.join(ROOT, "Output", month_day)

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")
    return directory_path


output_dir = get_directory()


def get_device() -> device:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA is available! Using GPU.')
    else:
        device = torch.device('cpu')
        print('CUDA is not available. Using CPU.')
    return device


device = get_device()


def plot_data_frequency(label_list: list[int], name_list: list[str] = None, tag: str = None) -> None:
    frequency_counter = Counter(label_list)
    sorted_items = sorted(frequency_counter.items(), key=lambda x: x[0])
    labels, counts = zip(*sorted_items)

    if name_list and max(labels) < len(name_list):
        labels = [name_list[label] for label in labels]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, counts)
    plt.xlabel(None)
    plt.ylabel('counts')
    plt.title(f'{tag if tag else "Dataset"}')
    plt.ylim(int(min(counts) * 0.85), int(max(counts) * 1.05))
    for index, count in enumerate(counts):
        plt.text(index, count + 0.1, str(count), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'class_frequency{" in " + tag if tag else ""}.png'))
    # plt.show()


def plot_accuracy(accuracies: Dict[str, List[float]]) -> None:
    plt.figure(figsize=(8, 5))

    epochs = range(1, len(accuracies[list(accuracies.keys())[0]]) + 1)
    for key, values in accuracies.items():
        plt.plot(epochs, values, marker='o', label=f'{key}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy on training and validation Dataset')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'accuracy_of_lstm.png'))
    # plt.show()
    return


def plot_confusion_matrix(confusion_matrix: np.ndarray, name_list: list[str]) -> None:
    """
    Plot the confusion matrix as a heatmap.

    Parameters:
    - confusion_matrix (np.ndarray): The confusion matrix.
    - output_size (int): Size of the output labels.

    Returns:
    - None
    """
    plt.figure(figsize=(12, 9))
    sns.heatmap(confusion_matrix, annot=True, fmt='.2f', cmap='Blues')
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    plt.title('Confusion Matrix')
    if len(name_list) == len(confusion_matrix):
        plt.xticks(ticks=[index + 0.5 for index in range(len(name_list))], labels=name_list)
        plt.yticks(ticks=[index + 0.5 for index in range(len(name_list))], labels=name_list)
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_of_lstm.png'))
    # plt.show()
    return


if __name__ == '__main__':
    plot_accuracy({"Training Dataset": [0.5, 0.4, 0.7, 0.3, 0.6, 0.3, 0.4, 0.8],
                   "Validation Dataset": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.3]})
