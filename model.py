import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import os
import shutil
import pickle
import cv2

IMAGE_WIDTH = 398
IMAGE_HEIGHT = 598
FRAME_COUNT = 4

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv3d(4, 16, (10, 10, 1))
        self.pool = nn.MaxPool3d((4, 4, 1), 2)
        self.conv2 = nn.Conv3d(16, 1, (10, 10, 1))
        self.fc1 = nn.Linear(12831, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if os.path.isfile("model"):
    shutil.copyfile("model", "old_model")
    model = torch.load("model")
else:
    model = NeuralNetwork()
    print("New model!")

loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=5e-4)

data_count = 0

def pour(frames, jumped):
    global data_count

    filename = f'tmp/data{data_count}'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as file:
        pickle.dump([frames, jumped], file)
    data_count += 1


def train(remove_last = 1):

    model.train()

    frames_train = []
    jumped_train = []

    jumped_count = 0

    for filename in os.listdir('tmp'):
        with open(f'tmp/{filename}', 'rb') as file:
            frames, jumped = pickle.load(file)
            frames_train.append(frames)
            jumped_train.append(jumped)
            jumped_count += 1 if jumped else 0

    shutil.rmtree('tmp')

    frames_train = frames_train[:len(frames_train)-remove_last]
    jumped_train = jumped_train[:len(jumped_train)-remove_last]

    print(len(frames_train), jumped_count / len(frames_train))

    dataset = []
    for X, y in zip(frames_train, jumped_train):
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
        y_tensor = torch.zeros(2)
        #if (y):
        #    y_tensor[1] = 1
        #else:
        #    y_tensor[0] = 0.6
        #    y_tensor[1] = 0.4
        y_tensor[int(y)] = 1
        dataset.append((X_tensor, y_tensor))

    loader = DataLoader(dataset, batch_size=16)

    running_loss = 0

    for data, labels in loader:
        optimizer.zero_grad()

        predict = model(data.to(device))

        loss = loss_fn(predict, labels.to(device))
        loss.backward()

        optimizer.step()
        
        running_loss += loss.item()

    print("Loss: ", running_loss)

def save():
    print(model.state_dict())
    torch.save(model, "model")


def should_jump(frames):

    model.eval()

    X_tensor = torch.tensor(np.array(frames), dtype=torch.float32)
    y_tensor = torch.zeros(2)

    dataset = [(X_tensor, y_tensor)]

    loader = DataLoader(dataset, batch_size=1)

    for data, label in loader:
        predict = model(data.to(device))
        print(predict, f"{predict[0][1].item()} > {predict[0][0].item()}", predict[0][1].item() > predict[0][0].item())
        return predict[0][1].item() > predict[0][0].item()
