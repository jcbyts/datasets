import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import h5py
from tqdm import tqdm
import os
from ..utils import create_time_embedding

def get_unit_ids(id=None):
    cids_list = {
        '20190120': [13, 16, 18, 20, 21, 22, 25, 27, 28, 29, 32, 33, 36, 47,61,64,65,66,68,78,79,80]
    }

    if id is None:
        for str in list(cids_list.keys()):
            print(str)
        return cids_list

    if id not in cids_list.keys():
        raise ValueError('Stimulus not found')
    
    return cids_list[id]

def get_stim_file(id=None):

    stim_list = {
            '20190120': 'Ellie_190120_0_0_30_30_1.mat'
        }

    if id is None:
        for str in list(stim_list.keys()):
            print(str)
        return stim_list

    if id not in stim_list.keys():
        raise ValueError('Stimulus not found')
    
    return stim_list[id]

class MTDotsDataset(Dataset):

    def __init__(self, sessname, dirname,
        num_lags = 18,
        download=False):

        self.sessname = sessname
        self.dirname = dirname
        self.filename = get_stim_file(sessname)
        self.cids = get_unit_ids(sessname)
        self.num_lags = num_lags
        # ensure_dir(self.dirname)

        # # check if we need to download the data
        # fpath = os.path.join(dirname, self.filename)
        # if not os.path.exists(fpath):
        #     print("File [%s] does not exist. Download set to [%s]" % (fpath, download))
        #     if download:
        #         print("Downloading set...")
        #         download_set(sessname, self.dirname)
        #     else:
        #         print("Download is False. Exiting...")
        #         return
        
        self.fhandle = h5py.File(os.path.join(self.dirname, self.filename), 'r')

        vel, Robs  = self.load_set()

        Xstim = create_time_embedding( vel, [self.num_lags, self.NX*self.num_channels, self.NY], tent_spacing=1)

        NC = len(self.cids)
        self.R = Robs[:,self.cids]
    
    def __getitem__(self, index):
        stim = self.Xstim[index,:]
        dfs = torch.ones(stim.shape, dtype=torch.float32)
        return {'stim': stim, 'robs': self.robs[index,:], 'dfs': dfs}

    def __len__(self) -> int:
        return self.NT

    def load_set(self):
        

        Robs = self.fhandle['MoStimY'][:,:].T
        X = self.fhandle['MoStimX'][:,:].T
        
        frameTime = X[:,0]

        # stim is NT x (NX*NY). Any non-zero value is the drift direction (as an integer) of a dot (at that spatial location)
        Stim = X[:,3:]

        #%% convert direciton stimulus to dx and dy
        # convert drift direction to degrees
        dbin = 360/np.max(Stim)

        # output will be x,y vectors
        dx = np.zeros(Stim.shape, dtype='float32')
        dy = np.zeros(Stim.shape, dtype='float32')

        stimind = np.where(Stim!=0) # non-zero values of Stim

        dx[stimind[0], stimind[1]]=np.cos(Stim[stimind[0], stimind[1]]*dbin/180*np.pi)
        dy[stimind[0], stimind[1]]=np.sin(Stim[stimind[0], stimind[1]]*dbin/180*np.pi)

        self.xax = np.arange(self.fhandle['GRID']['box'][0], self.fhandle['GRID']['box'][2], self.fhandle['GRID']['div'][0])
        self.yax = np.arange(self.fhandle['GRID']['box'][1], self.fhandle['GRID']['box'][3], self.fhandle['GRID']['div'][0])

        # concatenate dx/dy into one velocity stimulus
        self.NT = Stim.shape[0]
        self.NC = Robs.shape[1]
        self.NX = len(self.xax)
        self.NY = len(self.yax)

        vel = np.concatenate((dx, dy), axis=1)
        
        # weird python reshaping
        v_reshape = np.reshape(vel,[self.NT, 2, self.NX*self.NY])
        vel = np.transpose(v_reshape, (0,2,1)).reshape((self.NT, self.NX*self.NY*2))

        return vel, Robs

