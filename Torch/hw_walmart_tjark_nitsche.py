import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

class ClassificationDataSet(Dataset):
    def __init__(self, feature_file, label_file, train=True, test_split_ratio=0.3, transform=None, target_transform=None):

        self.features = pd.read_csv(feature_file)
        #print(self.labels.head())
        self.labels = self.features['Weekly_Sales']
        self.features.drop(['Weekly_Sales','Date'], inplace=True, axis= 1)
        #self.labels.drop(['Weekly_Sales','Date'], inplace= True, axis = 1)

        self.features = self.features.to_numpy()
        self.labels = self.labels.to_numpy()

        #print(self.features)
        #print(self.labels)

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
        self.linear_relu_stack = nn.Sequential(nn.Linear(in_features=15, out_features=500),nn.Sigmoid(),nn.Linear(500,1))

    def forward (self, x):
        logits = self.linear_relu_stack(x)
        return logits

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
                    loss, current = loss.item(),batch*len(X)
                    print(f"loss:{loss:>7f}[{current:>5d}/{size:>5d}]")
                    loss_history.append(loss)
        return loss_history


walmart_data_train = ClassificationDataSet('walmart_cleaned.csv','walmart_target.csv',train=True)
walmart_data_test = ClassificationDataSet('walmart_cleaned.csv','walmart_target.csv',train=False)

train_loader = torch.utils.data.DataLoader(walmart_data_train,batch_size=5000,shuffle=True)
test_loader = torch.utils.data.DataLoader(walmart_data_test,batch_size=1)

train_features, train_labels = next(iter(train_loader))

wm_model = NeuralNetwork()

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(wm_model.parameters(), lr=0.05)
wm_model.perform_training(train_loader, loss_fn, optimizer, epochs=10)
inputs = []
predictions = []
labels = []
for X, y in test_loader:
    labels.append(y.cpu().detach().numpy().squeeze())
    y_hat = wm_model(X)
predictions.append(y_hat.cpu().detach().numpy().squeeze())
plt.plot(np.arange(len(labels)), labels)
plt.plot(np.arange(len(predictions)), predictions)
plt.show()