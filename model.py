import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import math
import os
import shutil
import cv2

IMAGE_WIDTH = 400
IMAGE_HEIGHT = 600
FRAME_COUNT = 4


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        transforms = []

        inputs = IMAGE_WIDTH * IMAGE_HEIGHT * 3 * FRAME_COUNT
        while True:
            smaller = int(math.log2(inputs))
            if smaller <= 2:
                break

            transforms.append(nn.Linear(inputs, smaller))
            transforms.append(nn.ReLU())

            inputs = smaller

        transforms.append(nn.Linear(inputs, 2))

        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(*transforms)

    def forward(self, x):
        x = self.flatten(x)
        return self.stack(x)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if os.path.isfile("model"):
    shutil.copyfile("model", "old_model")
    model = torch.load("model")
    model.eval()
else:
    model = NeuralNetwork()
    print("New model!")

loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=5e-4)

frames_train = []
jumped_train = []


def pour(frames, jumped):
    frames_train.append(frames)
    jumped_train.append(jumped)


def remove_last(count):
    global frames_train
    global jumped_train

    frames_train = frames_train[:len(frames_train)-count]
    jumped_train = jumped_train[:len(jumped_train)-count]


def train():

    print(len(frames_train))

    dataset = []
    for X, y in zip(frames_train, jumped_train):
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
        y_tensor = torch.zeros(2)
        y_tensor[int(y)] = 1
        dataset.append((X_tensor, y_tensor))

    loader = DataLoader(dataset, batch_size=16)

    for data, labels in loader:
        predict = model(data.to(device))
        loss = loss_fn(labels.to(device), predict)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def save():
    print(model.state_dict())
    torch.save(model, "model")


def should_jump(frames):

    X_tensor = torch.tensor(np.array(frames), dtype=torch.float32)
    y_tensor = torch.zeros(2)

    dataset = [(X_tensor, y_tensor)]

    loader = DataLoader(dataset, batch_size=1)

    for data, label in loader:
        predict = model(data.to(device))
        print(predict, f"{predict[0][1].item()} < {predict[0][0].item()}", predict[0][1].item() < predict[0][0].item())
        return predict[0][1].item() < predict[0][0].item()
