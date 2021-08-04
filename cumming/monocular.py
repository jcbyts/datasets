
import os
import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import Dataset
import NDNT.NDNutils as NDNutils
from ..utils import download_file, ensure_dir


class MonocularDataset(Dataset):

    def __init__(self, sessname='expt04',
        num_lags=16,
        stimset='Train',
        dirname=None,
        corrected=True,
        download=True,
        device=None,
        dtype=torch.float32):
        """
        Monocular V1 1D noise bars
        Args:
            sessname: experiment name (default: 'expt04')
            dirname: where to look for data (THIS IS NECESSARY)
            num_lags: number of time lags to use (default: 16)
            stimset: which set of stimuli to load (default: 'Train')
            corrected: whether to use eye-tracking corrected stimulus (default: True)
            download: whether to download data if it is missing locally (default: True)
            device: device to load data onto (default: 'cpu')
            dtype: data type to load data in (default: torch.float32)
        """
        self.stimset = stimset
        self.corrected = corrected
        self.num_lags = num_lags
        self.device = device
        self.dtype = dtype
        self.dirname = dirname

        if dirname is None:
            raise ValueError('dirname must be specified')
        
        NDNutils.ensure_dir(dirname)

        # check if we need to download the data
        fpath = os.path.join(dirname, sessname + '.mat')
        if not os.path.exists(fpath):
            print("File [%s] does not exist. Download set to [%s]" % (fpath, download))
            if download:
                print("Downloading set...")
                download_set(sessname, self.dirname)
            else:
                print("Download is False. Exiting...")
                return

        stim, Robs, datafilters, Eadd_info = monocular_data_import( dirname, sessname, num_lags=num_lags )

        NX = stim.shape[1]
        Xstim = NDNutils.create_time_embedding( stim, [num_lags, NX, 1])
        NT, NC = Robs.shape
        # For index parsing
        used_inds = Eadd_info['used_inds']
        Ui, Xi = Eadd_info['TRinds'], Eadd_info['TEinds']
        print( "%d SUs, %d / %d used time points"%(NC, len(used_inds), NT) )
        
        # map numpy arrays into tensors
        if stimset=='Train':
            ix = Ui
        else:
            ix = Xi

        self.x = torch.tensor(Xstim[ix,:], dtype=dtype)
        self.y = torch.tensor(Robs[ix,:], dtype=dtype)
        if datafilters is not None:
            self.DFs = torch.tensor(datafilters[ix,:], dtype=dtype)
        else:
            self.DFs = torch.ones(Robs[ix,:].shape, dtype=torch.float32)
        
        if device:
            self.x =self.x.to(device)
            self.y =self.y.to(device)
            self.DFs =self.DFs.to(device)

        self.NC = Robs.shape[1]
        self.NX = NX
        self.NY = 1
        self.NF = 1
        
    def __getitem__(self, index):
        
        return {'stim': self.x[index,:], 'robs':self.y[index,:], 'dfs': self.DFs[index,:]}
        
    def __len__(self):
        return self.x.shape[0]



def get_stim_url(id):
    
    urlpath = {
            'expt01': 'https://www.dropbox.com/s/mn70kyohmp3kjnl/expt01.mat?dl=1',
            'expt02':'https://www.dropbox.com/s/pods4w89tbu2x57/expt02.mat?dl=1',
            'expt03': 'https://www.dropbox.com/s/p08375vcunrf9rh/expt03.mat?dl=1',
            'expt04': 'https://www.dropbox.com/s/zs1vcaz3sm01ncn/expt04.mat?dl=1',
            'expt05': 'https://www.dropbox.com/s/f3mpp3mlsrhof8k/expt05.mat?dl=1',
            'expt06': 'https://www.dropbox.com/s/saqjo7yibc6y8ut/expt06.mat?dl=1',
            'expt07': 'https://www.dropbox.com/s/op0rw7obzfvnm53/expt07.mat?dl=1',
            'expt08': 'https://www.dropbox.com/s/fwmtdegmlcdk9wo/expt08.mat?dl=1',
            'expt09': 'https://www.dropbox.com/s/yo8xo58ldiyrktm/expt09.mat?dl=1',
            'expt10': 'https://www.dropbox.com/s/k2zldzv7zfe7x06/expt10.mat?dl=1',
            'expt11': 'https://www.dropbox.com/s/rsc7h4njqntts39/expt11.mat?dl=1',
            'expt12': 'https://www.dropbox.com/s/yf1mm805j53yaj2/expt12.mat?dl=1',
            'expt13': 'https://www.dropbox.com/s/gidll8bgg5uie8h/expt13.mat?dl=1',
            'expt14': 'https://www.dropbox.com/s/kfof5m08g1v3rfe/expt14.mat?dl=1',
            'expt15': 'https://www.dropbox.com/s/zpcc7a2iy9bmkjd/expt15.mat?dl=1',
            'expt16': 'https://www.dropbox.com/s/b19kwdwy18d14hl/expt16.mat?dl=1',
        }
    
    if id not in urlpath.keys():
        raise ValueError('Stimulus URL not found')
    
    return urlpath[id]

def download_set(sessname, fpath):
    
    ensure_dir(fpath)

    # Download the data set
    url = get_stim_url(sessname)
    fout = os.path.join(fpath, sessname + '.mat')
    download_file(url, fout)

# --- Define data-helpers
def time_in_blocks(block_inds):
    num_blocks = block_inds.shape[0]
    #print( "%d number of blocks." %num_blocks)
    NT = 0
    for nn in range(num_blocks):
        NT += block_inds[nn,1]-block_inds[nn,0]+1
    return NT


def make_block_inds( block_lims, gap=20, separate = False):
    block_inds = []
    for nn in range(block_lims.shape[0]):
        if separate:
            block_inds.append(np.arange(block_lims[nn,0]-1+gap, block_lims[nn,1]), dtype='int')
        else:
            block_inds = np.concatenate( 
                (block_inds, np.arange(block_lims[nn,0]-1+gap, block_lims[nn,1], dtype='int')), axis=0)
    return block_inds


def monocular_data_import( datadir, exptn, num_lags=20 ):

    from copy import deepcopy

    time_shift = 1
    filename = exptn + '.mat'
    matdata = sio.loadmat( os.path.join(datadir,filename) )

    sus = matdata['goodSUs'][:,0] - 1  # switch from matlab indexing
    print('SUs:', sus)
    NC = len(sus)
    layers = matdata['layers'][0,:]
    block_list = matdata['block_inds'] # note matlab indexing
    stim_all = NDNutils.shift_mat_zpad(matdata['stimulus'], time_shift, 0)
    NTtot, NX = stim_all.shape
    DFs_all = deepcopy(matdata['data_filters'][:,sus])
    Robs_all = deepcopy(matdata['binned_SU'][:,sus])
    
    # Break up into train and test blocks
    # Assemble train and test indices based on BIlist
    NBL = block_list.shape[0]
    Xb = np.arange(2, NBL, 5)  # Every fifth trial is cross-validation
    Ub = np.array(list(set(list(range(NBL)))-set(Xb)), dtype='int')
    
    used_inds = make_block_inds( block_list, gap=num_lags )
    Ui, Xi = NDNutils.generate_xv_folds( len(used_inds) )
    TRinds, TEinds = used_inds[Ui].astype(int), used_inds[Xi].astype(int)

    Eadd_info = {
        'cortical_layer':layers, 'used_inds': used_inds, 
        'TRinds':TRinds, 'TEinds': TEinds, #'TRinds': Ui, 'TEinds': Xi, 
        'block_list': block_list, 'TRblocks': Ub, 'TEblocks': Xb}
    return stim_all, Robs_all, DFs_all, Eadd_info