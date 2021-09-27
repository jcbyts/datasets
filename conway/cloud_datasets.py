
import os
import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import Dataset
import NDNT.NDNutils as NDNutils
from ..utils import download_file, ensure_dir
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
        include_MUs = True,
        eyepos = None):

        self.datadir = datadir
        self.sess_list = sess_list
        self.num_lags = num_lags

        # get hdf5 file handles
        self.fhandles = [h5py.File(os.path.join(datadir, sess + '.hdf5'), 'r') for sess in self.sess_list]

        # build index map
        self.file_index = [] # which file the block corresponds to
        self.block_inds = []
        self.fix_n = []
        #self.unit_ids = []
        self.num_units = []
        self.num_sus = []
        self.NC = 0       
        self.stim_dims = None
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
        for f, fhandle in enumerate(self.fhandles):

            NSUfile = fhandle['Robs'].shape[1]
            NCfile = NSUfile
            if self.include_MUs:
                NCfile += fhandle['RobsMU'].shape[1]

            Nsac_file = fhandle['sac_inds'].shape[0]

            if self.stim_dims is None:
                self.stim_dims = fhandle['stim'].shape[1:]
            else:
                #check = self.stim_dims = fhandle['stim'].shape[1:]
                print('check dims: not implemented currently. Ignore')

            #self.unit_ids.append(self.NC + np.asarray(range(NCfile)))
            self.num_units.append(NCfile)
            self.num_sus.append(NSUfile)
            self.NC += NCfile

            sac_inds = fhandle['sac_inds']
            NStmp = sac_inds.shape[0]
            NT = fhandle['Robs'].shape[0]
            fix_count = 0
            # Break up by fixations based on sacc indices          
            for b in range(NStmp):
                self.file_index.append(f)
                
                if b < NStmp-1:
                    trange = np.arange(sac_inds[b]-1, sac_inds[b+1])
                else:
                    trange = np.arange(sac_inds[b]-1, NT)

                # Verify that there is some data there (rather than being a blank)
                if np.sum(fhandle['DFs']) > 0:
                    self.block_inds.append(deepcopy(trange))
                    self.fix_n.append(b+self.num_fixations)
                    fix_count += 1

            self.fixation_grouping.append(np.arange(fix_count, dtype='int64')+self.num_fixations)
            self.num_fixations += fix_count

        #self.dims = np.unique(np.asarray(self.dims)) # assumes they're all the same    
        if self.eyepos is not None:
            assert len(self.eyepos) == self.num_fixations, \
                "eyepos input should have %d fixations."%self.num_fixations

        # Develop default train, validation, and test datasets 
        #self.crossval_setup() 
    # END ColorClouds.__init__

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

    def __getitem__(self, index):
        
        if NDNutils.is_int(index):
            index = [index]
        elif type(index) is slice:
            index = list(range(index.start or 0, index.stop or len(self.block_inds), index.step or 1))

        stim = []
        robs = []
        dfs = []
        Xfix = []
        fixation_labels = []
        num_dims = self.stim_dims[0]*self.stim_dims[1]*self.stim_dims[2]
        
        for ii in index:
            inds = self.block_inds[ii]
            NT = len(inds)
            f = self.file_index[ii]
            fix_n = self.fix_n[ii]  # which fixation, across all datasets

            """ Stim """
            stim_tmp = torch.tensor(self.fhandles[f]['stim'][inds,:], dtype=torch.float32)
            if self.eyepos is not None:
                stim_tmp = self.shift_stim_fixation( stim_tmp, self.eyepos[fix_n] )

            # reshape and flatten stim: currently its NT x NX x NY x Nclrs
            stim_tmp = stim_tmp.permute([0,3,1,2]).reshape([-1, num_dims])

            """ Spikes: needs padding so all are B x NC """ 
            robs_tmp = torch.tensor(self.fhandles[f]['Robs'][inds,:], dtype=torch.float32)
            if self.include_MUs:
                robs_tmp = torch.cat(
                    (robs_tmp,
                    torch.tensor(self.fhandles[f]['RobsMU'][inds,:], dtype=torch.float32)), 
                    dim=1)
            NCbefore = int(np.asarray(self.num_units[:f]).sum())
            NCafter = int(np.asarray(self.num_units[f+1:]).sum())
            #robs_tmp = torch.cat(
            #    (torch.zeros( (NT, NCbefore), dtype=torch.float32),
            #    robs_tmp,
            #    torch.zeros( (NT, NCafter), dtype=torch.float32)),
            #    dim=1)

            """ Datafilters: needs padding like robs """
            dfs_tmp = torch.tensor(self.fhandles[f]['DFs'][inds,:], dtype=torch.float32)
            if self.include_MUs:
                dfs_tmp = torch.cat(
                    (dfs_tmp,
                    torch.tensor(self.fhandles[f]['DFsMU'][inds,:], dtype=torch.float32)),
                    dim=1)
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

