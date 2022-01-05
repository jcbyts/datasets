import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import h5py
import os
from .utils import get_stim_list, download_set
from datasets.mitchell.pixel.utils import shift_im

class Pixel(Dataset):
    '''
    Optional Covariates:
    'fixation_onset': onset of fixation
            'tent_ctrs' - array of tent centers
    'frame_tent':
            'ntents' - number of tent centers
        
    '''
    
    def __init__(self,
        dirname,
        sess_list:list=None,
        stimset:str="Train",
        requested_stims:list=["Gabor"],
        num_lags:int=12,
        load_shifters:bool=False,
        fixations_only:bool=False,
        valid_eye_rad:float=5.2,
        downsample_t:int=1,
        verbose=False,
        download=False,
        covariate_requests={}, # fixation_num, fixation_onset
        device=torch.device('cpu'),
    ):
    
        super().__init__()

        self.stimset = stimset
        self.requested_stims = requested_stims
        self.spike_sorting = 'kilowf' # only one option for now
        self.downsample_t = downsample_t
        self.num_lags = num_lags

        self.fixations_only = fixations_only
        if self.fixations_only:
            self.pad_fix_start = self.num_lags # pad by the number of stimulus lags

        self.valid_eye_rad = valid_eye_rad
        
        self.normalizing_constant = 50

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

                    # basic info
                    self.stim_indices[expt][stim]['frate'] = fhandle[stim][stimset]['Stim'].attrs['frate'][0]
                    self.stim_indices[expt][stim]['ppd'] = fhandle[stim][stimset]['Stim'].attrs['ppd'][0]
                    self.stim_indices[expt][stim]['rect'] = fhandle[stim][stimset]['Stim'].attrs['rect']
                    self.stim_indices[expt][stim]['center'] = fhandle[stim][stimset]['Stim'].attrs['center']

                    # valid indices
                    self.valid_idx.append(runninglength + self.get_valid_indices(self.fhandles[expt], expt, stim))

                    runninglength += sz[-1]
                    
                    self.dims[1] = np.maximum(self.dims[1], sz[0])
                    self.dims[2] = np.maximum(self.dims[1], sz[1])
        
        self.valid_idx = np.concatenate(self.valid_idx).astype(int)
        self.runninglength = runninglength
        self.dtype = torch.float32
        
        ''' Load data '''
        print("Loading data...")
        self.preload_numpy()

        # Handle requested covariates
        if 'fixation_num' in covariate_requests.keys():
            print("Loading fixation number covariate...")
            fix_inds = self.get_fixation_indices()
            self.covariates['fixation_num'] = np.zeros((self.covariates['stim'].shape[-1], 1))
            for num, fix in enumerate(fix_inds):
                self.covariates['fixation_num'][fix] = num
            print("Done")

        if 'fixation_onset' in covariate_requests.keys():
            print("Loading fixation onset covariate...")
            fix_inds = self.get_fixation_indices()
            fix_start = [fix[0] for fix in fix_inds]
            # nfix = len(fix_start)

            ctrs = covariate_requests['fixation_onset']['tent_ctrs']
            step = np.mean(np.diff(ctrs)/2)
            nlags = len(ctrs)

            sacstim = np.zeros((self.covariates['stim'].shape[-1], 1))
            sacstim[fix_start] = 1

            # make basis
            off = ctrs[0] - step
            if step >= 1:
                dt = 1 # assuming frame id instead of seconds
            else:
                dt = np.median(np.abs(np.diff(self.covariates['frame_times'], axis=0)))
            t = np.arange(off, ctrs[-1] + step, dt)
            B = np.maximum(0, 1-np.abs(t[:,None] - ctrs)/step)
            roll = int(np.round(off/dt))
            sacstim = np.roll(sacstim, roll, axis=0)

            # zero out invalid after shift
            if off < 0:
                sacstim[roll:,0] = 0
            elif off > 0:
                sacstim[:roll,0] = 0
                    
            # convolve with basis
            from scipy.signal import fftconvolve
            sacfull = fftconvolve(sacstim, B, mode="full")

            self.covariates['fixation_onset'] = sacfull[:self.covariates['stim'].shape[-1],:]
            self.fixation_onset_ctrs = ctrs
            print("Done")
            
        if 'frame_tent' in covariate_requests.keys():
            print("Loading frame tent covariate...")
            ntents = covariate_requests['frame_tent']['ntents']
            ctrs = np.linspace(self.covariates['frame_times'].min(), self.covariates['frame_times'].max(), ntents)
            step = np.mean(np.diff(ctrs)/2)
            nlags = len(ctrs)

            self.covariates['frame_tent'] = np.zeros((self.covariates['stim'].shape[-1], nlags))
            
            x = np.abs(self.covariates['frame_times'] - ctrs)
            x = np.maximum(0, ( 1-x/step ))
    
            self.covariates['frame_tent'] = x
            self.frame_tent_ctrs = ctrs
            print("Done")
            
            
        if device is not None:
            self.to_tensor(device)
            
        self.device = device
        if load_shifters:
            shifters = self.get_shifters(plot=True)
            self.correct_stim(shifters)
        
        self.shift = None
        self.raw_dims = self.dims

        self._crop_idx = [0, self.dims[1], 0, self.dims[2]]

    @property
    def num_lags(self):
        return self._num_lags//self.downsample_t
    
    @num_lags.setter
    def num_lags(self, value):
        self._num_lags = value*self.downsample_t

    @property
    def crop_idx(self):
        return self._crop_idx

    @crop_idx.setter
    def crop_idx(self, idx):
        self._crop_idx = idx
        self.dims[1] = self._crop_idx[1] - self._crop_idx[0]
        self.dims[2] = self._crop_idx[3] - self._crop_idx[2]

    def to_tensor(self, device):
        for cov in self.covariates.keys():
            self.covariates[cov] = torch.from_numpy(self.covariates[cov].astype('float32')).to(device)
        # self.stim = torch.tensor(self.stim, dtype=self.dtype, device=device)
        # self.robs = torch.tensor(self.robs.astype('float32'), dtype=self.dtype, device=device)
        # self.dfs = torch.tensor(self.dfs.astype('float32'), dtype=self.dtype, device=device)
        # self.eyepos = torch.tensor(self.eyepos.astype('float32'), dtype=self.dtype, device=device)
        # self.frame_times = torch.tensor(self.frame_times.astype('float32'), dtype=self.dtype, device=device)

    def preload_numpy(self):
        
        runninglength = self.runninglength
        ''' 
        Pre-allocate memory for data
        '''
        self.covariates = {'stim': np.zeros(  self.dims + [runninglength], dtype=np.float32),
                'robs': np.zeros(  [runninglength, self.NC], dtype=np.float32),
                'dfs': np.zeros(   [runninglength, self.NC], dtype=np.float32),
                'eyepos': np.zeros([runninglength, 2], dtype=np.float32),
                'frame_times': np.zeros([runninglength,1], dtype=np.float32)}

        for expt in self.sess_list:
            
            fhandle = self.fhandles[expt]

            for stim in self.requested_stims:
                if stim in fhandle.keys():
                    dt = 1/self.stim_indices[expt][stim]['frate']

                    sz = fhandle[stim][self.stimset]['Stim'].shape
                    inds = self.stim_indices[expt][stim]['inds']

                    self.covariates['stim'][0, :sz[0], :sz[1], inds] = np.transpose(fhandle[stim][self.stimset]['Stim'][...], [2,0,1])
                    self.covariates['frame_times'][inds] = fhandle[stim][self.stimset]['frameTimesOe'][...].T

                    """ EYE POSITION """
                    ppd = fhandle[stim][self.stimset]['Stim'].attrs['ppd'][0]
                    centerpix = fhandle[stim][self.stimset]['Stim'].attrs['center'][:]
                    eye_tmp = fhandle[stim][self.stimset]['eyeAtFrame'][1:3,:].T
                    eye_tmp[:,0] -= centerpix[0]
                    eye_tmp[:,1] -= centerpix[1]
                    eye_tmp/= ppd
                    self.covariates['eyepos'][inds,:] = eye_tmp

                    """ SPIKES """
                    frame_times = self.covariates['frame_times'][inds].flatten()

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

                    self.covariates['robs'][inds,:] = robs_tmp

                    """ DATAFILTERS """
                    unit_ids = self.spike_indices[expt]['unit ids']
                    for unit in unit_ids:
                        self.covariates['dfs'][inds, unit] = 1


        
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
                _ = plot_shifter(shifter, title=sess)
            
            shifters[sess] = shifter

        return shifters

    def correct_stim(self, shifters, verbose=True):
        if verbose:
            print("Correcting stim...")

        
        for sess in shifters.keys():
            if verbose:
                print("Correcting session [%s]" % sess)
            for stim in self.stim_indices[sess].keys():    
                if self.stim_indices[sess][stim]['corrected']:
                    print("Stim [%s] already corrected" % stim)
                else:
                    print("Correcting stim [%s]" % stim)
                    inds = self.stim_indices[sess][stim]['inds']

                    shift = shifters[sess](self.covariates['eyepos'][inds,:])
                    self.covariates['stim'][...,inds] = shift_im(self.covariates['stim'][...,inds].permute(3,0,1,2), shift).permute(1,2,3,0)
                    self.stim_indices[sess][stim]['corrected'] = True
        
        if verbose:
            print("Done correcting stim.")

    def get_valid_indices(self, fhandle, sess, stim):
        # get blocks (start, stop) of valid samples
        blocks = fhandle[stim][self.stimset]['blocks'][:,:]
        valid = []
        for bb in range(blocks.shape[1]):
            valid.append(np.arange(blocks[0,bb]+self.num_lags*self.downsample_t,
                blocks[1,bb])) # offset start by num_lags
        
        valid = np.concatenate(valid).astype(int)

        if self.fixations_only:
            # fixation inds
            fixations = fhandle[stim][self.stimset]['labels'][:].flatten()==1
            fix_starts = np.where(np.diff(fixations.astype('int8'))==1)[0]
            fix_stops = np.where(np.diff(fixations.astype('int8'))==-1)[0]
            if fixations[0]:
                fix_starts = np.insert(fix_starts, 0, 0)
            if fixations[-1]:
                fix_stops = np.append(fix_stops, fixations.size)
            nfix = fix_stops.size
            fixations = []
            for ifix in range(nfix):
                fix_inds = np.arange(fix_starts[ifix]+self.pad_fix_start, fix_stops[ifix])
                if len(fix_inds)>0:
                    fixations.append(fix_inds)
            fixations = np.concatenate(fixations)
            print("Found %d fixation indices" % fixations.size)
            print("%d valid indices" % valid.size)
            valid = np.intersect1d(valid, fixations)
            print("New valid size: %d" % valid.size)

            # fixations = np.where(fhandle[stim][self.stimset]['labels'][:]==1)[1]
            # valid = np.intersect1d(valid, fixations)
        
        if self.valid_eye_rad:
            ppd = self.stim_indices[sess][stim]['ppd']
            centerpix = self.stim_indices[sess][stim]['center']
            xy = fhandle[stim][self.stimset]['eyeAtFrame'][1:3,:].T
            xy[:,0] -= centerpix[0]
            xy[:,1] = centerpix[1] - xy[:,1] # y pixels run down (flip when converting to degrees)
            # convert to degrees
            xy = xy/ppd

            eyeCentered = np.hypot(xy[:,0],xy[:,1]) < self.valid_eye_rad
            valid = np.intersect1d(valid, np.where(eyeCentered)[0])

        return valid
    
    def get_fixation_indices(self, index_valid=False):
        fixations = []
        for sess in self.sess_list:
            for stim in self.stim_indices[sess].keys():
                for ii in range(len(self.stim_indices[sess][stim]['fix_start'])):
                    fix_inds = np.arange(self.stim_indices[sess][stim]['fix_start'][ii], 
                    self.stim_indices[sess][stim]['fix_stop'][ii])
                    if index_valid:
                        fix_inds = np.where(np.in1d(self.valid_idx, fix_inds))[0]
                    fixations.append(fix_inds)
        
        return fixations
    
    def get_train_indices(self, seed=1234, frac_train=0.8):

        fixations = self.get_fixation_indices()
        nfix = len(fixations)
        np.random.seed(seed)
        train_fix = np.random.choice(nfix, size=int(frac_train*nfix), replace=False)

        fixes = np.asarray(fixations, dtype=object)

        train_inds = np.concatenate(fixes[train_fix])

        train_inds = np.where(np.in1d(self.valid_idx, train_inds))[0]
        val_inds = np.setdiff1d(np.arange(len(self.valid_idx)), train_inds).tolist()
        train_inds = train_inds.tolist()

        return train_inds, val_inds
    
    def get_stas(self, train_loader=None, device=None, batch_size=1000, inds=None, square=False):
        from torch.utils.data import DataLoader, Subset
        from tqdm import tqdm
        
        if train_loader is None:
            if inds is None:
                train_loader = DataLoader(self, batch_size=batch_size, shuffle=False, num_workers=int(os.cpu_count()//2))
            else:
                ds = Subset(self, inds)
                train_loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=int(os.cpu_count()//2))

        if device is None:
            device = torch.device('cpu')

        xy = 0
        ny = 0
        for data in tqdm(train_loader):
            x = torch.flatten(data['stim'], start_dim=1)
            if square:
                x=x**2
            y = data['robs']*data['dfs']
            x = x.to(device)
            y = y.to(device)
        
            xy += (x.T@y).detach().cpu()
            ny += (y.sum(dim=0)).detach().cpu()

        stas = (xy/ny).reshape(self.dims[1:] + [self.num_lags] + [self.NC]).permute(2,0,1,3)

        return stas

    def compute_datafilters(self, batch_size=1000, sd_thresh=2):
        
        import os
        r = [] # m
        rsd = []
        ft = []
        ftend = []

        NT = self.covariates['robs'].shape[0]
        nbatch = NT // batch_size
        r = []
        for i in range(nbatch):
            rtmp = self.covariates['robs'][range(i*batch_size, i*batch_size+batch_size),:]
            r.append(rtmp.mean(dim=0).detach().numpy())
            rsd.append(rtmp.std(dim=0).detach().numpy())
            ft.append(self.covariates['frame_times'][i*batch_size].detach().numpy())
            ftend.append(self.covariates['frame_times'][i*batch_size+batch_size].detach().numpy())

        ft = np.asarray(ft).flatten()
        ftend = np.asarray(ftend).flatten()
        r = np.asarray(r)
        rsd = np.asarray(rsd)

        z0 = np.mean(rsd, axis=0)
        z1 = np.std(rsd, axis=0)
        zs = (rsd - z0)/z1

        dfs = np.logical_and(zs > -sd_thresh, zs < sd_thresh)
        dfs[np.sum(dfs, axis=1)<.9*rsd.shape[1],:] = False
        dfs[r==0] = False

        bad_epochs_start = []
        bad_epochs_stop = []
        for cc in range(self.NC):
            ii = np.where(~dfs[:,cc])[0]
            
            bad_epochs_start.append(ft[ii])
            bad_epochs_stop.append(ftend[ii])

        frame_times = self.covariates['frame_times'].clone()

        big_dfs = self.covariates['dfs'].clone()
        for cc in range(self.NC):
            for epoch_start, epoch_stop in zip(bad_epochs_start[cc], bad_epochs_stop[cc]):
                
                bad = np.where(np.logical_and(frame_times > epoch_start, frame_times < epoch_stop))[0]
                big_dfs[bad,cc]=0
        
        self.covariates['dfs'] = big_dfs
        # remove bad valid indices
        iix = np.where((big_dfs[self.valid_idx,:].sum(dim=1)==0).numpy())[0]
        self.valid_idx = np.delete(self.valid_idx, iix)


    def __len__(self):
        return len(self.valid_idx)
    
    def __getitem__(self, idx):
        
        if self.shift is not None: # crop stimulus after shifting (slower if no shifting)
            if self.downsample_t > 1:
                s = 0
                for t in range(self.downsample_t):
                    s += self.covariates['stim'][...,self.valid_idx[idx,None]-np.arange(t, self._num_lags, self.downsample_t)]/self.normalizing_constant
                s = s/self.downsample_t
            else:
                s = self.covariates['stim'][..., self.valid_idx[idx,None]-range(self.num_lags)]/self.normalizing_constant
                
        else: # crop stimulus on load
            if self.downsample_t > 1:
                s = 0
                for t in range(self.downsample_t):
                    s += self.covariates['stim'][:,self.crop_idx[0]:self.crop_idx[1], self.crop_idx[2]:self.crop_idx[3],self.valid_idx[idx,None]-np.arange(t, self._num_lags, self.downsample_t)]/self.normalizing_constant
                s = s/self.downsample_t
            else:
                s = self.covariates['stim'][:,self.crop_idx[0]:self.crop_idx[1], self.crop_idx[2]:self.crop_idx[3],self.valid_idx[idx,None]-range(self._num_lags)]/self.normalizing_constant
        
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s.astype('float32'))
        
        if len(s.shape)==5: # handle array vs singleton indices
                s = s.permute(3,0,1,2,4) # N, C, H, W, T
        
        if self.shift is not None:
            sz = list(s.shape)
            if self.downsample_t > 1:
                lagged_shift = 0
                for t in range(self.downsample_t):
                    lagged_shift = self.shift[self.valid_idx[idx,None]-np.arange(t, self._num_lags, self.downsample_t),:]
                lagged_shift /= self.downsample_t
                lagged_shift = lagged_shift.reshape([-1, 2])
            else:
                lagged_shift = self.shift[self.valid_idx[idx,None]-range(self.num_lags),:].reshape([-1,2])

            if len(sz)==5:
                s = shift_im(s.permute(3,4,0,1,2).reshape([-1] + sz[1:4]),
                        lagged_shift).reshape([-1] + [self.num_lags] + sz[1:4]).permute(0,2,3,4,1)
            else:
                s = shift_im(s.permute(3,0,1,2).reshape([-1] + sz[0:3]),
                        self.shift[self.valid_idx[idx,None]-range(self.num_lags),:].reshape([-1,2])).reshape([self.num_lags] + sz[0:3]).permute(1,2,3,0)
            
            # apply crop after shifting
            s = s[...,self.crop_idx[0]:self.crop_idx[1], self.crop_idx[2]:self.crop_idx[3], :]

        out = {'stim': s} # stim has already been processed
        covs = list(self.covariates.keys())
        covs.remove('stim')
        for cov in covs:
            out[cov] = self.covariates[cov][self.valid_idx[idx],:]

        return out