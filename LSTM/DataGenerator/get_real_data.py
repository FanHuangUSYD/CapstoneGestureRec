import glob
import os
import warnings
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from LSTM.util import DEVICE


def transform_single_yolov5_video_output(txt_file_path: str, class_label: int, total_frames: int, box_num: int = 2) -> tuple[list[list[float]], list[int]]:
    x_train, y_train = [], []
    txt_files = glob.glob(txt_file_path + "/*.txt")
    if not txt_files:
        warnings.warn(f'{txt_file_path} has no txt files')
        return x_train, y_train

    # get video filename
    prefix = os.path.basename(txt_files[0]).split('_')[0]

    for frame in range(total_frames):
        file_path = os.path.join(txt_file_path, f'{prefix}_{frame}.txt')
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                content = file.read()
                content = content.rstrip()
                # TODO
                # need to filter specific gesture tag
                content = [float(i) for i in content.replace('\n', ' ').split(" ")]
                if len(content) != box_num * 2:
                    warnings.warn(f'{txt_file_path} output {file_path} frame has abnormal {len(content)//6} boxes')
                if len(content) == box_num * 2:
                    x_train.append(content[:box_num*6])
                    y_train.append(class_label)
        else:
            # need to be adjusted
            x_train.append([-1.0]*box_num*6)
            # 0 means no gesture detected
            y_train.append(0)

    return x_train, y_train


def get_lstm_train_data(data_desc: dict[str, Tuple[int, int]], batch_size: int=10) -> DataLoader[tuple[Tensor, ...]]:
    """
    Args:
        data_desc: map video file path to action label and frame number
    eg: {'DATA/OUTPUT/video1/exp/best/labels': (1, 300)}
    Returns:
        Dataloader
    """
    x_train, y_train = [], []
    for video_file_path, (label, frame_number) in data_desc.items():
        sub_train, sub_y = transform_single_yolov5_video_output(video_file_path, label, frame_number)
        x_train.extend(sub_train)
        y_train.extend(sub_y)

    feature, label = np.array(x_train).astype(np.float32), np.array(y_train).astype(np.int64)
    device = DEVICE
    dataset = TensorDataset(torch.from_numpy(feature).to(device), torch.from_numpy(label).to(device))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


if __name__ == "__main__":
    # print(transform_single_yolov5_video_output("../../YOLOv5/DATA/OUTPUT/video1/exp/best/labels", 1, 300))
    print(get_lstm_train_data(
        {
            '../../YOLOv5/DATA/OUTPUT/video12/exp/best/labels': (1, 2000),
            '../../YOLOv5/DATA/OUTPUT/video13/exp/best/labels': (2, 1030),
        }
    ))
