
import os
import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import Dataset
import NDNT.NDNutils as NDNutils
from ..utils import download_file, ensure_dir
from copy import deepcopy
import h5py
class MultiDataset(Dataset):
    """
    MULTIDATASET can load batches from multiple datasets
    """

    def __init__(self,
        sess_list,
        dirname, 
        num_lags=1):

        self.dirname = dirname
        self.sess_list = sess_list
        self.num_lags = num_lags

        # get hdf5 file handles
        self.fhandles = [h5py.File(os.path.join(dirname, sess + '.hdf5'), 'r') for sess in self.sess_list]

        # build index map
        self.file_index = [] # which file the block corresponds to
        self.block_inds = []

        self.unit_ids = []
        self.num_units = []
        self.NC = 0       
        self.dims = []

        for f, fhandle in enumerate(self.fhandles):
            NCfile = fhandle['robs'].shape[1]
            self.dims.append(fhandle['stim'].shape[1])
            self.unit_ids.append(self.NC + np.asarray(range(NCfile)))
            self.num_units.append(NCfile)
            self.NC += NCfile

            NT = fhandle['robs'].shape[0]

            blocks = (np.sum(fhandle['dfs'][:,:], axis=1)==0).astype(np.float32)
            blocks[0] = 1 # set invalid first sample
            blocks[-1] = 1 # set invalid last sample

            blockstart = np.where(np.diff(blocks)==-1)[0]
            blockend = np.where(np.diff(blocks)==1)[0]
            nblocks = len(blockstart)

            for b in range(nblocks):
                self.file_index.append(f)
                self.block_inds.append(np.arange(blockstart[b], blockend[b]))

        self.dims = np.unique(np.asarray(self.dims)) # assumes they're all the same

    def __getitem__(self, index):
        
        if type(index) is int:
            index = [index]
        elif type(index) is slice:
            index = list(range(index.start or 0, index.stop or len(self.block_inds), index.step or 1))

        stim = []
        robs = []
        dfs = []
        for ii in index:
            inds = self.block_inds[ii]
            NT = len(inds)
            f = self.file_index[ii]

            """ Stim """
            stim_tmp = torch.tensor(self.fhandles[f]['stim'][inds,:], dtype=torch.float32)

            """ Spikes: needs padding so all are B x NC """ 
            robs_tmp = torch.tensor(self.fhandles[f]['robs'][inds,:], dtype=torch.float32)
            NCbefore = int(np.asarray(self.num_units[:f]).sum())
            NCafter = int(np.asarray(self.num_units[f+1:]).sum())
            robs_tmp = torch.cat(
                (torch.zeros( (NT, NCbefore), dtype=torch.float32),
                robs_tmp,
                torch.zeros( (NT, NCafter), dtype=torch.float32)),
                dim=1)

            """ Datafilters: needs padding like robs """
            dfs_tmp = torch.tensor(self.fhandles[f]['dfs'][inds,:], dtype=torch.float32)
            dfs_tmp[:self.num_lags,:] = 0 # invalidate the filter length
            dfs_tmp = torch.cat(
                (torch.zeros( (NT, NCbefore), dtype=torch.float32),
                dfs_tmp,
                torch.zeros( (NT, NCafter), dtype=torch.float32)),
                dim=1)

            stim.append(stim_tmp)
            robs.append(robs_tmp)
            dfs.append(dfs_tmp)

        stim = torch.cat(stim, dim=0)
        robs = torch.cat(robs, dim=0)
        dfs = torch.cat(dfs, dim=0)

        return {'stim': stim, 'robs': robs, 'dfs': dfs}
        

    def __len__(self):
        return len(self.block_inds)


class MultiDatasetFix(Dataset):
    """
    MULTIDATASET-FIX can load batches from multiple datasets
    Two changes from regular MultiDataset (above):
    -- technical: uses now-present 'block_inds' variable
    -- adds fixation-number-Xstim to get_item
    Might be more clever way rather than duplicating so much code: replace?
    -- if replace, can have a flag about whether to calc Xfix?

    Constructor will take eye position, which for now is an input from data
    generated in the session (not on disk). It should have the length size 
    of the total number of fixations x1.
    """

    def __init__(self,
        sess_list,
        dirname, 
        num_lags=16,
        eyepos = None):

        self.dirname = dirname
        self.sess_list = sess_list
        self.num_lags = num_lags

        # get hdf5 file handles
        self.fhandles = [h5py.File(os.path.join(dirname, sess + '.hdf5'), 'r') for sess in self.sess_list]

        # build index map
        self.file_index = [] # which file the block corresponds to
        self.block_inds = []
        self.fixation_number = []
        self.unit_ids = []
        self.num_units = []
        self.NC = 0       
        self.dims = []
        self.eyepos = eyepos

        self.num_fixations = 0
        for f, fhandle in enumerate(self.fhandles):
            NCfile = fhandle['robs'].shape[1]
            self.dims.append(fhandle['stim'].shape[1])
            self.unit_ids.append(self.NC + np.asarray(range(NCfile)))
            self.num_units.append(NCfile)
            self.NC += NCfile

            NT = fhandle['robs'].shape[0]

            #blocks = (np.sum(fhandle['dfs'][:,:], axis=1)==0).astype(np.float32)
            #blocks[0] = 1 # set invalid first sample
            #blocks[-1] = 1 # set invalid last sample

            #blockstart = np.where(np.diff(blocks)==-1)[0]
            #blockend = np.where(np.diff(blocks)==1)[0]
            blockstart = fhandle['block_inds'][:,0]
            blockend = fhandle['block_inds'][:,1]
            nblocks = len(blockstart)

            for b in range(nblocks):
                self.file_index.append(f)
                self.block_inds.append(np.arange(blockstart[b], blockend[b]))
                self.fixation_number.append(b+self.num_fixations)

            self.num_fixations += nblocks

        self.dims = np.unique(np.asarray(self.dims)) # assumes they're all the same    
        if self.eyepos is not None:
            assert len(self.eyepos) == self.num_fixations, \
                "eyepos input should have %d fixations."%self.num_fixations
    # END MultiDatasetFix.__init__

    def shift_stim_fixation( self, stim, shift):
        """Simple shift by integer (rounded shift) and zero padded. Assumes 
        the stim is a full fixation, to be shifted by same amount."""
        sh = round(shift)
        shstim = stim.new_zeros(*stim.shape)
        if sh < 0:
            shstim[:, -sh:] = stim[:, :sh]
        elif sh > 0:
            shstim[:, :-sh] = stim[:, sh:]
        else:
            shstim = deepcopy(stim)

        return shstim

    def __getitem__(self, index):
        
        if type(index) is int:
            index = [index]
        elif type(index) is slice:
            index = list(range(index.start or 0, index.stop or len(self.block_inds), index.step or 1))

        stim = []
        robs = []
        dfs = []
        Xfix = []
        for ii in index:
            inds = self.block_inds[ii]
            NT = len(inds)
            f = self.file_index[ii]
            fix_num = self.fixation_number[ii]

            """ Stim """
            stim_tmp = torch.tensor(self.fhandles[f]['stim'][inds,:], dtype=torch.float32)
            if self.eyepos is not None:
                stim_tmp = self.shift_stim_fixation( stim_tmp, self.eyepos[fix_num] )

            """ Spikes: needs padding so all are B x NC """ 
            robs_tmp = torch.tensor(self.fhandles[f]['robs'][inds,:], dtype=torch.float32)
            NCbefore = int(np.asarray(self.num_units[:f]).sum())
            NCafter = int(np.asarray(self.num_units[f+1:]).sum())
            robs_tmp = torch.cat(
                (torch.zeros( (NT, NCbefore), dtype=torch.float32),
                robs_tmp,
                torch.zeros( (NT, NCafter), dtype=torch.float32)),
                dim=1)

            """ Datafilters: needs padding like robs """
            dfs_tmp = torch.tensor(self.fhandles[f]['dfs'][inds,:], dtype=torch.float32)
            dfs_tmp[:self.num_lags,:] = 0 # invalidate the filter length
            dfs_tmp = torch.cat(
                (torch.zeros( (NT, NCbefore), dtype=torch.float32),
                dfs_tmp,
                torch.zeros( (NT, NCafter), dtype=torch.float32)),
                dim=1)

            """ Fixation Xstim """
            # Do we need fixation-Xstim or simply index for array? Let's assume Xstim
            fix_tmp = torch.zeros( (NT, self.num_fixations), dtype=torch.float32)
            fix_tmp[:, fix_num] = 1.0

            stim.append(stim_tmp)
            robs.append(robs_tmp)
            dfs.append(dfs_tmp)
            Xfix.append(fix_tmp)

        stim = torch.cat(stim, dim=0)
        robs = torch.cat(robs, dim=0)
        dfs = torch.cat(dfs, dim=0)
        Xfix = torch.cat(Xfix, dim=0)

        return {'stim': stim, 'robs': robs, 'dfs': dfs, 'Xfix': Xfix}

    def __len__(self):
        return len(self.block_inds)

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