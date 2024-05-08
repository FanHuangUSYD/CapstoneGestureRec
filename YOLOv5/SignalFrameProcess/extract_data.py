'''
This file contains all tools which extracting key data from a single frame via YOLOv5 Model
'''

import os

import torch
from torch import Tensor
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import random


def show_extracted_data(extracted_data: Tensor, frame: np.ndarray) -> None:
    data_list = extracted_data.tolist()
    text_lines = [" ".join(f"{num:.2f}" for num in sublist) for sublist in data_list]
    # text_to_display = "\n".join(text_lines)
    text_start_x = 10
    text_start_y = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)    # white
    line_type = 2

    for i, line in enumerate(text_lines):
        y = text_start_y + i * 30  # 更新每行文本的垂直位置
        cv2.putText(frame, line, (text_start_x, y), font, font_scale, font_color, line_type)

    # cv2.putText(frame, text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Display the image with text
    cv2.imshow("Image with Text", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def extract_frame(model: torch.nn.Module, frame: np.ndarray) -> Tensor:
    # Process the frame data with model
    results = model(frame)
    # features = results.pandas().xyxy[0]
    features = results.xyxy[0]
    #Convert the list data into a tuple for standardization.
    #features_tuple = tuple(features)
    return features


if __name__ == "__main__":
    # locate the root directory of YOLOv5 folder
    current_directory = os.getcwd()
    YOLOv5_directory = os.path.dirname(current_directory)
    # initialize the model
    model_path = os.path.join(YOLOv5_directory, "DATA", "developData", "testing_model.pt")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    if os.path.isfile(model_path):
        # Load local model if exist
        model.load_state_dict(torch.load(model_path))
        print("Local model found.\n Existing model loaded.\n")
    else:
        # otherwise load from torch hub
        torch.save(model.state_dict(), model_path)
        print("Saveing model to local directory.\n")
    '''
    All of the testing resources will be stored in developData
    '''
    # load a testing image
    example_path = os.path.join(YOLOv5_directory, "DATA", "developData", "zidane.jpg")
    test_frame = cv2.imread(example_path)

    data = extract_frame(model, test_frame)
    show_extracted_data(data, test_frame)

