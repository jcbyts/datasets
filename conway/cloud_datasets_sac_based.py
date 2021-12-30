
import os
import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import Dataset
import NDNT.utils as utils
#from NDNT.utils import download_file, ensure_dir
from copy import deepcopy
import h5py


class ColorClouds(Dataset):
    """
    -- can load batches from multiple datasets
    -- hdf5 files must have the following information:
        Robs
        RobsMU
        stim: 4-d stimulus: time x nx x ny x color
        block_inds: start and stop of 'trials' (perhaps fixations for now)
        other things: saccades? or should that be in trials? 

    Constructor will take eye position, which for now is an input from data
    generated in the session (not on disk). It should have the length size 
    of the total number of fixations x1.
    """

    def __init__(self,
        sess_list,
        datadir, 
        num_lags=12,
        include_MUs = False,
        preload = True,
        time_embed = 2,  # 0 is no time embedding, 1 is time_embedding with get_item, 2 is pre-time_embedded
        eyepos = None, 
        device=torch.device('cpu')):
        """Constructor options"""

        self.datadir = datadir
        self.sess_list = sess_list
        self.device = device

        self.num_lags = num_lags
        if time_embed == 2:
            assert preload, "Cannot pre-time-embed without preloading."
        self.preload = preload
        self.time_embed = time_embed

        # get hdf5 file handles
        self.fhandles = [h5py.File(os.path.join(datadir, sess + '.hd5'), 'r') for sess in self.sess_list]

        # build index map
        self.data_threshold = 6  # how many valid time points required to include saccade?
        self.file_index = [] # which file the block corresponds to
        self.block_inds = []
        self.fix_n = []
        #self.unit_ids = []
        self.num_units = []
        self.num_sus = []
        self.NC = 0       
        #self.stim_dims = None
        self.eyepos = eyepos
        self.generate_Xfix = False
        self.fixation_grouping = []
        self.include_MUs = include_MUs
        self.SUinds = []
        self.MUinds = []

        # Set up to store default train_, val_, test_inds
        self.test_inds = None
        self.val_inds = None
        self.train_inds = None

        self.num_fixations = 0
        runninglength = 0

        for f, fhandle in enumerate(self.fhandles):

            NSUfile = fhandle['Robs'].shape[1]
            NCfile = NSUfile
            if self.include_MUs:
                NCfile += fhandle['RobsMU'].shape[1]

            Nsac_file = fhandle['sac_inds'].shape[0]

            #if self.stim_dims is None:
            self.dims = [fhandle['stim'].shape[3]] + list(fhandle['stim'].shape[1:3]) + [1]
            #else:
                #check = self.stim_dims = fhandle['stim'].shape[1:]
            #    print('check dims: not implemented currently. Use data default. Ignoring stim_dims argument.')
            
            if self.time_embed > 0:
                self.dims[3] = self.num_lags

            #self.unit_ids.append(self.NC + np.asarray(range(NCfile)))
            self.num_units.append(NCfile)
            self.num_sus.append(NSUfile)
            self.NC += NCfile

            sac_inds = fhandle['sac_inds']
            NStmp = sac_inds.shape[0]
            NT = fhandle['Robs'].shape[0]
            runninglength += NT

            fix_count = 0

            # Break up by fixations based on sacc indices          
            for b in range(NStmp):
                self.file_index.append(f)
                
                if b < NStmp-1:
                    trange = list(np.arange(sac_inds[b]-1, sac_inds[b+1]))
                else:
                    trange = list(np.arange(sac_inds[b]-1, NT))

                # Verify that there is some data there (rather than being a blank)
                if np.mean(np.sum(fhandle['DFs'][trange[num_lags:], :],axis=0)) > self.data_threshold:
                    self.block_inds.append(deepcopy(trange))
                    self.fix_n.append(b+self.num_fixations)
                    fix_count += 1

            self.fixation_grouping.append(np.arange(fix_count, dtype='int64')+self.num_fixations)
            self.num_fixations += fix_count

        self.runninglength  = runninglength

        #self.dims = np.unique(np.asarray(self.dims)) # assumes they're all the same    
        if self.eyepos is not None:
            assert len(self.eyepos) == self.num_fixations, \
                "eyepos input should have %d fixations."%self.num_fixations

        if preload:
            print("Loading data into memory...")
            self.preload_numpy()

            if time_embed == 2:
                print("Time embedding...")
                idx = np.arange(runninglength)
                self.stim = np.transpose(
                    self.stim[np.arange(runninglength)[:,None]-np.arange(num_lags), :, :, :],
                    axes=[0,2,3,4,1])

            # Flatten stim 
            self.stim = np.reshape(self.stim, [runninglength, -1])

            # Convert data to tensors
            #if self.device is not None:
            self.to_tensor(self.device)
            print("Done.")

        # Develop default train, validation, and test datasets 
            #self.crossval_setup() 
    # END ColorClouds.__init__

    def preload_numpy(self):
        """Note this loads stimulus but does not time-embed"""

        NT = self.runninglength
        ''' 
        Pre-allocate memory for data
        '''
        self.stim = np.zeros( [NT] + self.dims[:3], dtype=np.float32)
        self.robs = np.zeros( [NT, self.NC], dtype=np.float32)
        self.dfs = np.ones( [NT, self.NC], dtype=np.float32)
        #self.eyepos = np.zeros([NT, 2], dtype=np.float32)
        #self.frame_times = np.zeros([NT,1], dtype=np.float32)

        t_counter = 0
        unit_counter = 0
        for ee in range(len(self.fhandles)):
            
            fhandle = self.fhandles[ee]
            sz = fhandle['stim'].shape
            inds = range(t_counter, t_counter+sz[0])
            #inds = self.stim_indices[expt][stim]['inds']
            self.stim[inds, ...] = np.transpose( np.array(fhandle['stim'], dtype=np.float32), axes=[0,3,1,2])
            #self.frame_times[inds] = fhandle[stim][self.stimset]['frameTimesOe'][...].T

            """ EYE POSITION """
            #ppd = fhandle[stim][self.stimset]['Stim'].attrs['ppd'][0]
            #centerpix = fhandle[stim][self.stimset]['Stim'].attrs['center'][:]
            #eye_tmp = fhandle[stim][self.stimset]['eyeAtFrame'][1:3,:].T
            #eye_tmp[:,0] -= centerpix[0]
            #eye_tmp[:,1] -= centerpix[1]
            #eye_tmp/= ppd
            #self.eyepos[inds,:] = eye_tmp

            """ SPIKES """            
            #frame_times = self.frame_times[inds].flatten()

            #spike_inds = np.where(np.logical_and(
            #    fhandle['Neurons'][self.spike_sorting]['times']>=frame_times[0],
            #    fhandle['Neurons'][self.spike_sorting]['times']<=frame_times[-1]+0.01)
            #    )[1]

            #st = fhandle['Neurons'][self.spike_sorting]['times'][0,spike_inds]
            #clu = fhandle['Neurons'][self.spike_sorting]['cluster'][0,spike_inds].astype(int)
            # only keep spikes that are in the requested cluster ids list

            #ix = np.in1d(clu, self.spike_indices[expt]['unit ids orig'])
            #st = st[ix]
            #clu = clu[ix]
            # map cluster id to a unit number
            #clu = self.spike_indices[expt]['unit ids map'][clu]

            #robs_tmp = torch.sparse_coo_tensor( np.asarray([np.digitize(st, frame_times)-1, clu]),
            #    np.ones(len(clu)), (len(frame_times), self.NC) , dtype=torch.float32)
            #robs_tmp = robs_tmp.to_dense().numpy().astype(np.int8)
            
            #discontinuities = np.diff(frame_times) > 1.25*dt
            #if np.any(discontinuities):
            #    print("Removing discontinuitites")
            #    good = np.where(~discontinuities)[0]
            #    robs_tmp = robs_tmp[good,:]
            #    inds = inds[good]

            """ Robs and DATAFILTERS"""
            robs_tmp = np.zeros( [len(inds), self.NC], dtype=np.float32 )
            dfs_tmp = np.zeros( [len(inds), self.NC], dtype=np.float32 )
            num_sus = fhandle['Robs'].shape[1]
            units = range(unit_counter, unit_counter+num_sus)
            robs_tmp[:, units] = np.array(fhandle['Robs'], dtype=np.float32)
            dfs_tmp[:, units] = np.array(fhandle['DFs'], dtype=np.float32)
            if self.include_MUs:
                num_mus = fhandle['RobsMU'].shape[1]
                units = range(unit_counter+num_sus, unit_counter+num_sus+num_mus)
                robs_tmp[:, units] = np.array(fhandle['RobsMU'], dtype=np.float32)
                dfs_tmp[:, units] = np.array(fhandle['DFsMU'], dtype=np.float32)
            
            self.robs[inds,:] = deepcopy(robs_tmp)
            self.dfs[inds,:] = deepcopy(dfs_tmp)

            """ DATAFILTERS """

            #unit_ids = self.spike_indices[expt]['unit ids']
            #for unit in unit_ids:
            #    self.dfs[inds, unit] = 1
            t_counter += sz[0]
            unit_counter += self.num_units[ee]

    # END .preload_numpy()

    def to_tensor(self, device):
        self.stim = torch.tensor(self.stim, dtype=torch.float32, device=device)
        self.robs = torch.tensor(self.robs, dtype=torch.float32, device=device)
        self.dfs = torch.tensor(self.dfs, dtype=torch.float32, device=device)
        #self.eyepos = torch.tensor(self.eyepos.astype('float32'), dtype=self.dtype, device=device)
        #self.frame_times = torch.tensor(self.frame_times.astype('float32'), dtype=self.dtype, device=device)

    def shift_stim_fixation( self, stim, shift):
        """Simple shift by integer (rounded shift) and zero padded. Note that this is not in 
        is in units of number of bars, rather than -1 to +1. It assumes the stim
        has a batch dimension (over a fixation), and shifts the whole stim by the same amount."""
        print('Currently needs to be fixed to work with 2D')
        sh = round(shift)
        shstim = stim.new_zeros(*stim.shape)
        if sh < 0:
            shstim[:, -sh:] = stim[:, :sh]
        elif sh > 0:
            shstim[:, :-sh] = stim[:, sh:]
        else:
            shstim = deepcopy(stim)

        return shstim
    # END MultiDatasetFix.shift_stim_fixation

    def crossval_setup(self, folds=5, random_gen=False, test_set=True):
        """This sets the cross-validation indices up We can add featuers here. Many ways to do this
        but will stick to some standard for now. It sets the internal indices, which can be read out
        directly or with helper functions. Perhaps helper_functions is the best way....
        
        Inputs: 
            random_gen: whether to pick random fixations for validation or uniformly distributed
            test_set: whether to set aside first an n-fold test set, and then within the rest n-fold train/val sets
        Outputs:
            None: sets internal variables test_inds, train_inds, val_inds
        """

        test_fixes = []
        tfixes = []
        vfixes = []
        for ee in range(len(self.fixation_grouping)):
            fix_inds = self.fixation_grouping[ee]
            vfix1, tfix1 = self.fold_sample(len(fix_inds), folds, random_gen=random_gen)
            if test_set:
                test_fixes += list(fix_inds[vfix1])
                vfix2, tfix2 = self.fold_sample(len(tfix1), folds, random_gen=random_gen)
                vfixes += list(fix_inds[tfix1[vfix2]])
                tfixes += list(fix_inds[tfix1[tfix2]])
            else:
                vfixes += list(fix_inds[vfix1])
                tfixes += list(fix_inds[tfix1])

        self.val_inds = np.array(vfixes, dtype='int64')
        self.train_inds = np.array(tfixes, dtype='int64')
        if test_set:
           self.test_inds = np.array(test_fixes, dtype='int64')
    # END MultiDatasetFix.crossval_setup

    def fold_sample( self, num_items, folds, random_gen=False):
        """This really should be a general method not associated with self"""
        if random_gen:
            num_val = int(num_items/folds)
            tmp_seq = np.random.permutation(num_items)
            val_items = np.sort(tmp_seq[:num_val])
            rem_items = np.sort(tmp_seq[num_val:])
        else:
            offset = int(folds//2)
            val_items = np.arange(offset, num_items, folds, dtype='int32')
            rem_items = np.delete(np.arange(num_items, dtype='int32'), val_items)
        return val_items, rem_items

    def get_max_samples(self, gpu_n=0, history_size=1, nquad=0, num_cells=None, buffer=1.2):
        """
        get the maximum number of samples that fit in memory -- for GLM/GQM x LBFGS

        Inputs:
            dataset: the dataset to get the samples from
            device: the device to put the samples on
        """
        if gpu_n == 0:
            device = torch.device('cuda:0')
        else:
            device = torch.device('cuda:1')

        if num_cells is None:
            num_cells = self.NC
        
        t = torch.cuda.get_device_properties(device).total_memory
        r = torch.cuda.memory_reserved(device)
        a = torch.cuda.memory_allocated(device)
        free = t - (a+r)

        data = self[0]
        mempersample = data['stim'].element_size() * data['stim'].nelement() + 2*data['robs'].element_size() * data['robs'].nelement()
    
        mempercell = mempersample * (nquad+1) * (history_size + 1)
        buffer_bytes = buffer*1024**3

        maxsamples = int(free - mempercell*num_cells - buffer_bytes) // mempersample
        print("# samples that can fit on device: {}".format(maxsamples))
        return maxsamples
    # END .get_max_samples

    def __getitem__(self, index):
        
        if utils.is_int(index):
            index = [index]
        elif type(index) is slice:
            index = list(range(index.start or 0, index.stop or len(self.block_inds), index.step or 1))

        stim = []
        robs = []
        dfs = []
        Xfix = []
        fixation_labels = []
        #num_dims = self.stim_dims[0]*self.stim_dims[1]*self.stim_dims[2]
        num_dims = self.dims[0]*self.dims[1]*self.dims[2]
        
        for ii in index:
            inds = self.block_inds[ii]
            NT = len(inds)
            fix_n = self.fix_n[ii]  # which fixation, across all datasets

            if self.preload:
                stim_tmp = self.stim[inds,:]
                robs_tmp = self.robs[inds,:]
                dfs_tmp = self.dfs[inds,:]

            else:
                f = self.file_index[ii]
                """ Stim """
                stim_tmp = torch.tensor(self.fhandles[f]['stim'][inds,:], dtype=torch.float32)
                # reshape and flatten stim: currently its NT x NX x NY x Nclrs
                stim_tmp = stim_tmp.permute([0,3,1,2]).reshape([-1, num_dims])

                """ Spikes: needs padding so all are B x NC """ 
                robs_tmp = torch.tensor(self.fhandles[f]['Robs'][inds,:], dtype=torch.float32)
                if self.include_MUs:
                    robs_tmp = torch.cat(
                        (robs_tmp,
                        torch.tensor(self.fhandles[f]['RobsMU'][inds,:], dtype=torch.float32)), 
                        dim=1)

                """ Datafilters: needs padding like robs """
                dfs_tmp = torch.tensor(self.fhandles[f]['DFs'][inds,:], dtype=torch.float32)
                if self.include_MUs:
                    dfs_tmp = torch.cat(
                        (dfs_tmp,
                        torch.tensor(self.fhandles[f]['DFsMU'][inds,:], dtype=torch.float32)),
                        dim=1)

            """ Additional processing within fixation"""
            """ Stim """
            if self.eyepos is not None:
                stim_tmp = self.shift_stim_fixation( stim_tmp, self.eyepos[fix_n] )

            """ Spikes: needs padding so all are B x NC """ 
            #NCbefore = int(np.asarray(self.num_units[:f]).sum())
            #NCafter = int(np.asarray(self.num_units[f+1:]).sum())
            #robs_tmp = torch.cat(
            #    (torch.zeros( (NT, NCbefore), dtype=torch.float32),
            #    robs_tmp,
            #    torch.zeros( (NT, NCafter), dtype=torch.float32)),
            #    dim=1)

            dfs_tmp[:self.num_lags,:] = 0 # invalidate the filter length
            #dfs_tmp = torch.cat(
            #    (torch.zeros( (NT, NCbefore), dtype=torch.float32),
            #    dfs_tmp,
            #    torch.zeros( (NT, NCafter), dtype=torch.float32)),
            #    dim=1)

            """ Fixation Xstim """
            # Do we need fixation-Xstim or simply index for array? Let's assume Xstim
            if self.generate_Xfix:
                fix_tmp = torch.zeros( (NT, self.num_fixations), dtype=torch.float32)
                fix_tmp[:, fix_n] = 1.0
                Xfix.append(fix_tmp)
            else:
                #fix_tmp = torch.ones(NT, dtype=torch.float32) * fix_n
                #fixation_labels.append(fix_tmp.int())
                fixation_labels.append(torch.ones(NT, dtype=torch.int64) * fix_n)
    
            stim.append(stim_tmp)
            robs.append(robs_tmp)
            dfs.append(dfs_tmp)

        stim = torch.cat(stim, dim=0)
        robs = torch.cat(robs, dim=0)
        dfs = torch.cat(dfs, dim=0)

        if self.generate_Xfix:
            Xfix = torch.cat(Xfix, dim=0)
            return {'stim': stim, 'robs': robs, 'dfs': dfs, 'Xfix': Xfix}
        else:
            fixation_labels = torch.cat(fixation_labels, dim=0)
            return {'stim': stim, 'robs': robs, 'dfs': dfs, 'fix_n': fixation_labels}

    def __len__(self):
        return len(self.block_inds)

