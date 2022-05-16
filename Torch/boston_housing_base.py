import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd



class ClassificationDataSet(Dataset):
    def __init__(self, feature_file, label_file, train=True, test_split_ratio=0.3, transform=None, target_transform=None):
        self.features = pd.read_csv(feature_file).to_numpy()
        self.labels = pd.read_csv(label_file).to_numpy()

        self.features = (self.features - np.min(self.features, axis=0)) / (
                    np.max(self.features, axis=0) - np.min(self.features, axis=0))
        self.labels = (self.labels - np.min(self.labels, axis=0)) / (
                    np.max(self.labels, axis=0) - np.min(self.labels, axis=0))

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        self.features = torch.from_numpy(self.features).float().to(device)
        self.labels = torch.from_numpy(self.labels).float().to(device)

        dataset_size = len(self.labels)
        indices = list(range(dataset_size))
        split = int(np.floor(test_split_ratio * dataset_size))
        np.random.seed(100)
        np.random.shuffle(indices)
        train_indices, test_indices = indices[split:], indices[:split]
        if train:
            self.features = self.features[train_indices]
            self.labels = self.labels[train_indices]
        else:
            self.features = self.features[test_indices]
            self.labels = self.labels[test_indices]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.labels[idx]
        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            label = self.target_transform(label)
        return features, label

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        #self.flatten =nn.Flatten()
        self.linear_relu_stack = nn.Sequential(nn.Linear(in_features=13, out_features=20),nn.Sigmoid(),nn.Linear(20,1))

    def forward (self, x):
        logits = self.linear_relu_stack(x)
        return logits

    """def perform_training(self, X, y, dataloader):
        loss_fn = nn.MSELoss
        optimizer = torch.optim.SGD(self.parameters(), lr = 0.05)
        for (X, y) in dataloader:
            # Compute prediction and loss
            pred = self(X)
            loss = loss_fn(pred, y)
            # Backpropagation
            optimizer.zero_grad
            loss.backward
            optimizer.step"""

    def perform_training(self, dataloader,loss_fn,optimizer,epochs=100):
        loss_history = []
        for t in range(epochs):
            print(f"Epoch{t+1}\n-------------------")
            size = len(dataloader.dataset)
            for batch,(X,y) in enumerate(dataloader):
                # Compute prediction and loss
                pred = self(X)
                loss = loss_fn(pred, y)
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch % 1 == 0:
                    loss, current = loss.item(),batch*len(x)
                    print(f"loss:{loss:>7f}[{current:>5d}/{size:>5d}]")
                    loss_history.append(loss)
        return loss_history

training_data = ClassificationDataSet("BostonTarget.csv","BostonFeature.csv",train=True)
test_data = ClassificationDataSet("BostonTarget.csv","BostonFeature.csv",train=False)

train_dataloader = DataLoader(training_data , batch_size =64, shuffle =True)
test_dataloader = DataLoader(test_data , batch_size =64, shuffle =True)
train_features, train_labels = next(iter(train_dataloader))

model = NeuralNetwork()
history = model.perform_training(train_dataloader,nn.MSELoss,torch.optim.SGD(model.parameters(), lr = 0.05))
plt.plot(history)
plt.show()
