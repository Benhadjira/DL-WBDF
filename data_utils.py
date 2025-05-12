import torch
from torch.utils.data import Dataset
import numpy as np

def prepare_data(weak_patches, strong_patches):
    weak_labels = np.zeros(len(weak_patches)) 
    strong_labels = np.ones(len(strong_patches))

    data = np.concatenate([weak_patches, strong_patches], axis=0)
    labels = np.concatenate([weak_labels, strong_labels], axis=0)
    return data, labels

class PatchDataset(Dataset):
    def __init__(self, data, labels, normalize=True):
        self.data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
        if normalize:
            mean = self.data.mean()
            std = self.data.std()
            self.data = (self.data - mean) / std
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        return self.data[idx], self.labels[idx]
