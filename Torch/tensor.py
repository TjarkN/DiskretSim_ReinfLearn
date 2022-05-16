import os

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class BostonFeatureDataset(Dataset):

    def __init__(self, annotations_file, file_dir, transform = None, target_transform = None):
        self.labels = pd.read_csv(annotations_file)
        self.file = pd.read_csv(file_dir)
        self.file_dir = file_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        #item_path = os.path.join(self.file_dir,self.labels.iloc[idx, 0])
        #item = pd.read_csv(item_path)
        item = self.file.iloc[idx]
        label = self.labels.iloc[idx,0]
        if self.transform:
            item = self.transform(item)
        if self.target_transform:
            label = self.target_transform(label)
        return item, label

data_set = BostonFeatureDataset("BostonTarget.csv","BostonFeature.csv")
print(data_set.__getitem__(2))

dataloader = DataLoader(data_set,batch_size=64,shuffle = True)


