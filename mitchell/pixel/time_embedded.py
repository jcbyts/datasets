import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import h5py
import os
from .utils import get_stim_list, download_set
from datasets.mitchell.pixel.utils import shift_im

class Pixel(Dataset):
    
    def __init__(self,
        dirname,
        sess_list:list=None,
        stimset:str="Train",
        requested_stims:list=["Gabor"],
        num_lags:int=12,
        load_shifters:bool=False,
        cids:list=None,
        flatten:bool=False,
        dim_order:str='cwht',
        fixations_only:bool=False,
        valid_eye_rad:float=5.2,
        downsample_t:int=1,
        verbose=False,
        download=False,
        device=torch.device('cpu'),
    ):

        super().__init__()

        self.stimset = stimset
        self.requested_stims = requested_stims
        self.spike_sorting = 'kilowf' # only one option for now
        self.num_lags = num_lags

        self.fixations_only = fixations_only
        self.valid_eye_rad = False
        self.downsample_t = downsample_t

        # find valid sessions
        stim_list = get_stim_list() # list of valid sessions
        new_sess = []
        for sess in sess_list:
            if sess in stim_list.keys():
                if verbose:
                    print("Found [%s]" %sess)
                new_sess.append(sess)

        self.dirname = dirname
        self.sess_list = new_sess # is a list of valid sessions
        self.fnames = [get_stim_list(sess) for sess in self.sess_list] # is a list of filenames
        self.download_stim_files(download) # download the files
        # open hdf5 files as a list of handles
        self.fhandles = {sess: h5py.File(os.path.join(dirname, get_stim_list(sess)), 'r') for sess in self.sess_list}

        runninglength = 0
        self.dims = [1, 0, 0]
        self.NC = 0

        self.valid_idx = []

        self.spike_indices = {}
        self.stim_indices = {}
        
        '''
        Loop over all sessions, get valid indices, neurons, stimulus size, etc.
        '''
        for expt in self.sess_list:

            fhandle = self.fhandles[expt]    
            
            '''
            SPIKES
            '''
            unique_units = fhandle['Neurons'][self.spike_sorting]['cids'][0,:].astype(int)
            
            # map unit ids to index into the new ids
            mc = np.max(unique_units)+1
            unit_map = -1*np.ones(mc, dtype=int)
            unit_map[unique_units] = np.arange(len(unique_units))+self.NC
            
            # number of units in this session
            num_units = len(unique_units)

            self.spike_indices[expt] = {'unit ids orig': unique_units,
                'unit ids map': unit_map, 
                'unit ids': np.arange(self.NC, num_units+self.NC)}

            self.NC += num_units
            self.stim_indices[expt] = {}

            '''
            STIM / BEHAVIOR Indices
            '''
            for stim in self.requested_stims:
                if stim in self.fhandles[expt].keys():
                    sz = fhandle[stim][stimset]['Stim'].shape
                    
                    self.stim_indices[expt][stim] = {'inds': np.arange(runninglength, runninglength + sz[-1]), 'corrected': False}
                    self.valid_idx.append(runninglength + self.get_valid_indices(self.fhandles[expt], stim))

                    # stats
                    self.stim_indices[expt][stim]['frate'] = fhandle[stim][stimset]['Stim'].attrs['frate'][0]
                    self.stim_indices[expt][stim]['ppd'] = fhandle[stim][stimset]['Stim'].attrs['ppd'][0]
                    self.stim_indices[expt][stim]['rect'] = fhandle[stim][stimset]['Stim'].attrs['rect']

                    # fixation inds
                    fixations = fhandle[stim][self.stimset]['labels'][:].flatten()==1
                    fix_starts = np.where(np.diff(fixations.astype('int8'))==1)[0]
                    fix_stops = np.where(np.diff(fixations.astype('int8'))==-1)[0]
                    if fixations[0]:
                        fix_starts = np.insert(fix_starts, 0, 0)
                    if fixations[-1]:
                        fix_stops = np.append(fix_stops, fixations.size)

                    self.stim_indices[expt][stim]['fix_start'] = fix_starts + runninglength
                    self.stim_indices[expt][stim]['fix_stop'] = fix_stops + runninglength

                    runninglength += sz[-1]
                    
                    self.dims[1] = np.maximum(self.dims[1], sz[0])
                    self.dims[2] = np.maximum(self.dims[1], sz[1])
        
        self.valid_idx = np.concatenate(self.valid_idx)
        self.runninglength = runninglength
        self.dtype = torch.float32
        
        ''' Load data '''
        print("Loading data...")
        self.preload_numpy()

        if device is not None:
            self.to_tensor(device)
            
        self.device = device
        if load_shifters:
            shifters = self.get_shifters()
            self.correct_stim(shifters)
        
        self.shift = None
        self.raw_dims = self.dims

        self._crop_idx = [0, self.dims[1], 0, self.dims[2]]
        
    @property
    def crop_idx(self):
        return self._crop_idx

    @crop_idx.setter
    def crop_idx(self, idx):
        self._crop_idx = idx
        self.dims[1] = self._crop_idx[1] - self._crop_idx[0]
        self.dims[2] = self._crop_idx[3] - self._crop_idx[2]

    def to_tensor(self, device):
        self.stim = torch.tensor(self.stim, dtype=self.dtype, device=device)
        self.robs = torch.tensor(self.robs.astype('float32'), dtype=self.dtype, device=device)
        self.dfs = torch.tensor(self.dfs.astype('float32'), dtype=self.dtype, device=device)
        self.eyepos = torch.tensor(self.eyepos.astype('float32'), dtype=self.dtype, device=device)
        self.frame_times = torch.tensor(self.frame_times.astype('float32'), dtype=self.dtype, device=device)

    def preload_numpy(self):
        
        runninglength = self.runninglength
        ''' 
        Pre-allocate memory for data
        '''
        self.stim = np.zeros(  self.dims + [runninglength], dtype=np.int8)
        self.robs = np.zeros(  [runninglength, self.NC], dtype=np.int8)
        self.dfs = np.zeros(   [runninglength, self.NC], dtype=np.int8)
        self.eyepos = np.zeros([runninglength, 2], dtype=np.float32)
        self.frame_times = np.zeros([runninglength,1], dtype=np.float32)

        for expt in self.sess_list:
            
            fhandle = self.fhandles[expt]

            for stim in self.requested_stims:
                if stim in fhandle.keys():
                    dt = 1/self.stim_indices[expt][stim]['frate']

                    sz = fhandle[stim][self.stimset]['Stim'].shape
                    inds = self.stim_indices[expt][stim]['inds']

                    self.stim[0, :sz[0], :sz[1], inds] = np.transpose(fhandle[stim][self.stimset]['Stim'][...], [2,0,1])
                    self.frame_times[inds] = fhandle[stim][self.stimset]['frameTimesOe'][...].T

                    """ EYE POSITION """
                    ppd = fhandle[stim][self.stimset]['Stim'].attrs['ppd'][0]
                    centerpix = fhandle[stim][self.stimset]['Stim'].attrs['center'][:]
                    eye_tmp = fhandle[stim][self.stimset]['eyeAtFrame'][1:3,:].T
                    eye_tmp[:,0] -= centerpix[0]
                    eye_tmp[:,1] -= centerpix[1]
                    eye_tmp/= ppd
                    self.eyepos[inds,:] = eye_tmp

                    """ SPIKES """
                    frame_times = self.frame_times[inds].flatten()

                    spike_inds = np.where(np.logical_and(
                        fhandle['Neurons'][self.spike_sorting]['times']>=frame_times[0],
                        fhandle['Neurons'][self.spike_sorting]['times']<=frame_times[-1]+0.01)
                        )[1]

                    st = fhandle['Neurons'][self.spike_sorting]['times'][0,spike_inds]
                    clu = fhandle['Neurons'][self.spike_sorting]['cluster'][0,spike_inds].astype(int)
                    # only keep spikes that are in the requested cluster ids list

                    ix = np.in1d(clu, self.spike_indices[expt]['unit ids orig'])
                    st = st[ix]
                    clu = clu[ix]
                    # map cluster id to a unit number
                    clu = self.spike_indices[expt]['unit ids map'][clu]

                    robs_tmp = torch.sparse_coo_tensor( np.asarray([np.digitize(st, frame_times)-1, clu]),
                        np.ones(len(clu)), (len(frame_times), self.NC) , dtype=torch.float32)
                    robs_tmp = robs_tmp.to_dense().numpy().astype(np.int8)
                    
                    discontinuities = np.diff(frame_times) > 1.25*dt
                    if np.any(discontinuities):
                        print("Removing discontinuitites")
                        good = np.where(~discontinuities)[0]
                        robs_tmp = robs_tmp[good,:]
                        inds = inds[good]

                    self.robs[inds,:] = robs_tmp

                    """ DATAFILTERS """
                    unit_ids = self.spike_indices[expt]['unit ids']
                    for unit in unit_ids:
                        self.dfs[inds, unit] = 1


        
    def download_stim_files(self, download=True):
        if not download:
            return

        # check if files exist. download if they don't
        for isess,fname in enumerate(self.fnames):
            fpath = os.path.join(self.dirname, fname)
            if not os.path.exists(fpath):
                print("File [%s] does not exist. Download set to [%s]" % (fpath, download))
                if download:
                    print("Downloading set...")
                    download_set(self.sess_list[isess], self.dirname)
                else:
                    print("Download is False. Exiting...")

    def get_shifters(self, plot=False):

        shifters = {}
        for sess in self.sess_list:
            print("Checking for shifters for session [%s]" % sess)
            sfname = [f for f in os.listdir(self.dirname) if 'shifter_' + sess in f]
                
            if len(sfname) == 0:
                from datasets.mitchell.pixel.utils import download_shifter
                download_shifter(sess, self.dirname)
            else:
                print("Shifter exists")
                import pickle
                fname = os.path.join(self.dirname, sfname[0])
                print("Loading shifter from %s" % fname)
                shifter_res = pickle.load(open(fname, "rb"))
                shifter = shifter_res['shifters'][np.argmin(shifter_res['vallos'])]

            if plot:
                from datasets.mitchell.pixel.utils import plot_shifter
                _ = plot_shifter(shifter)
            
            shifters[sess] = shifter

        return shifters

    def correct_stim(self, shifters, verbose=True):
        if verbose:
            print("Correcting stim...")

        
        from tqdm import tqdm
        for sess in tqdm(shifters.keys()):
            if verbose:
                print("Correcting session [%s]" % sess)
            for stim in self.stim_indices[sess].keys():    
                if self.stim_indices[sess][stim]['corrected']:
                    print("Stim [%s] already corrected" % stim)
                else:
                    print("Correcting stim [%s]" % stim)
                    inds = self.stim_indices[sess][stim]['inds']

                    shift = shifters[sess](self.eyepos[inds,:])
                    self.stim[...,inds] = shift_im(self.stim[...,inds].permute(3,0,1,2), shift).permute(1,2,3,0)
                    self.stim_indices[sess][stim]['corrected'] = True
        
        if verbose:
            print("Done correcting stim.")

    def __len__(self):
        return len(self.valid_idx)
    
    def __getitem__(self, idx):
                    
        s = self.stim[..., self.valid_idx[idx,None]-range(self.num_lags)]
        
        if not self.device:
            s = torch.tensor(s.astype('float32'))
        
        if len(s.shape)==5: # handle array vs singleton indices
                s = s.permute(3,0,1,2,4) # N, C, H, W, T
        
        if self.shift is not None:
            if len(s.shape)==5:
                s = shift_im(s.permute(3,4,0,1,2).reshape([-1] + self.rawdims),
                        self.shift[self.valid_idx[idx,None]-range(self.num_lags),:].reshape([-1,2])).reshape([-1] + [self.num_lags] + self.dims).permute(0,2,3,4,1)
            else:
                s = shift_im(s.permute(3,0,1,2).reshape([-1] + self.rawdims),
                        self.shift[self.valid_idx[idx,None]-range(self.num_lags),:].reshape([-1,2])).reshape([self.num_lags] + self.dims).permute(1,2,3,0)

        s = s[...,self.crop_idx[0]:self.crop_idx[1], self.crop_idx[2]:self.crop_idx[3], :]

        if self.device:
            out = {'stim': s,
                'robs': self.robs[self.valid_idx[idx],:],
                'dfs': self.dfs[self.valid_idx[idx],:],
                'eyepos': self.eyepos[self.valid_idx[idx],:],
                'frame_times': self.frame_times[self.valid_idx[idx],:]}
        else:

            out = {'stim': s,
                'robs': torch.tensor(self.robs[self.valid_idx[idx],:].astype('float32')),
                'dfs': torch.tensor(self.dfs[self.valid_idx[idx],:].astype('float32')),
                'eyepos': torch.tensor(self.eyepos[self.valid_idx[idx],:].astype('float32')),
                'frame_times': torch.tensor(self.frame_times[self.valid_idx[idx],:].astype('float32'))}

        return out
    
    def get_valid_indices(self, fhandle, stim):
        # get blocks (start, stop) of valid samples
        blocks = fhandle[stim][self.stimset]['blocks'][:,:]
        valid = []
        for bb in range(blocks.shape[1]):
            valid.append(np.arange(blocks[0,bb]+self.num_lags*self.downsample_t,
                blocks[1,bb])) # offset start by num_lags
        
        valid = np.concatenate(valid).astype(int)

        if self.fixations_only:
            fixations = np.where(fhandle[stim][self.stimset]['labels'][:]==1)[1]
            valid = np.intersect1d(valid, fixations)
        
        if self.valid_eye_rad:
            xy = fhandle[stim][self.stimset]['eyeAtFrame'][1:3,:].T
            xy[:,0] -= self.centerpix[0]
            xy[:,1] = self.centerpix[1] - xy[:,1] # y pixels run down (flip when converting to degrees)
            # convert to degrees
            xy = xy/self.ppd
            # subtract offset
            xy[:,0] -= self.valid_eye_ctr[0]
            xy[:,1] -= self.valid_eye_ctr[1]
            eyeCentered = np.hypot(xy[:,0],xy[:,1]) < self.valid_eye_rad
            valid = np.intersect1d(valid, np.where(eyeCentered)[0])

        return valid