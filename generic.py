
import torch
from torch.utils.data import Dataset


class GenericDataset(Dataset):

    def __init__(self,
        stim,
        robs,
        dfs=None,
        device=None):

        self.stim = stim
        self.robs = robs
        self.dfs = dfs

        if device is None:
            device = torch.device('cpu')
        
        self.device = device

        if len(stim.shape) > 3:
            self.stim = self.stim.contiguous(memory_format=torch.channels_last)

        self.stim = self.stim.to(device)
        self.robs = self.robs.to(device)
        self.dfs = self.dfs.to(device)
        
        
    def __len__(self):

        return self.stim.shape[0]

    def __getitem__(self, index):
        return {'stim': self.stim[index,...],
            'robs': self.robs[index,...],
            'dfs': self.dfs[index,...]}