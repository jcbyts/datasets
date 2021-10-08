
import os
import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import Dataset
import NDNT.NDNutils as NDNutils
from ..utils import download_file, ensure_dir
from copy import deepcopy
import h5py


class WhiskerData(Dataset):
    """
    -- can load batches from multiple datasets IN principle, but many variables not set up to do so
    -- hdf5 files must have the following information:
        [Response info]
        Robs
        [stim info]
        
        [trial information]
        block_inds: start and stop of 'trials' 

    """

    def __init__(self,
        sess_list,
        datadir, 
        num_lags=20,
        hemis=0,  # hemis should be 0 (L), 1 (R), or 2 (L+R): full produced anyway
        num_phase_bins = 12,
        LVhemis = None,
        LVsmooth = 1,
        numLVlags = 1,
        preload = True):
        # preload currently not implemented = False option

        assert preload, 'havent implemented non-preloaded. may never have to?'
        assert len(sess_list) == 1, 'Cant yet do multiple datasets.'
        self.preload = preload
        self.datadir = datadir
        self.sess_list = sess_list
        self.num_lags = num_lags
        self.trial_grouping = []   # trial list associated with each file
    
        # get hdf5 file handles
        self.fhandles = [h5py.File(os.path.join(datadir, sess + '.hd5'), 'r') for sess in self.sess_list]

        # Which hemispheres
        self.hemis = hemis
        self.LVhemis = LVhemis
        self.LVsmooth = LVsmooth
        
        # build index map
        self.file_index = [] # which file the block corresponds to
        self.block_inds = []
        #self.unit_ids = []
        self.num_units = []
        self.NC = 0       
        self.generate_Xphase = False

        # Set up to store default train_, val_, test_inds
        self.test_inds = None
        self.val_inds = None
        self.train_inds = None
        self.num_cells = [0, 0]

        self.num_trials = 0
        for f, fhandle in enumerate(self.fhandles):

            NT = fhandle['touches'].shape[0]
            NCLfile, NCRfile = fhandle['num_cells']
            TRinds = fhandle['TRinds']
            Ntr_file = TRinds.shape[0]

            #self.unit_ids.append(self.NC + np.asarray(range(NCfile)))
            self.num_units.append([NCLfile, NCRfile])
            self.num_cells[0] += NCLfile
            self.num_cells[1] += NCRfile
            self.use_units = []

            if self.preload:
                self.touches_full = torch.zeros([NT, 4], dtype=torch.float32)
                self.RobsL = torch.zeros([NT, NCLfile], dtype=torch.float32)
                self.RobsR = torch.zeros([NT, NCRfile], dtype=torch.float32)

            # these dont depend on preloading because they are just used in constructor (for now)
            # and they are by trial so easy to just read into memory
            self.TRpistons = self.fhandles[f]['TRpistons']
            phases = torch.zeros([NT, 4], dtype=torch.float32)

            tr_count = 0
            # Organize trials
            for b in range(Ntr_file):
                self.file_index.append(f)
                
                trange = np.arange(TRinds[b, 0], TRinds[b, 1])
                # Verify that there is some data there (rather than being a blank)
                # BUT CURRENTLY NO DATAFILTERS
                #if np.sum(fhandle['DFs'][trange[num_lags:],:]) > 0:
                self.block_inds.append(deepcopy(trange))
                #    #self.fix_n.append(b+self.num_fixations)

                # Loading data bit-by-bit is better than all at once
                phases[trange, :] = torch.tensor(
                    self.fhandles[f]['phases'][trange,:], 
                    dtype=torch.float32)
                    
                if self.preload:
                    self.touches_full[trange, :] = torch.tensor(
                        self.fhandles[f]['touches'][trange, :], 
                        dtype=torch.float32)

                    self.RobsL[trange, :] = torch.tensor(
                        self.fhandles[f]['RobsL'][trange, :].astype(float), 
                        dtype=torch.float32)

                    self.RobsR[trange, :] = torch.tensor(
                        self.fhandles[f]['RobsR'][trange, :].astype(float), 
                        dtype=torch.float32)

                tr_count += 1

            self.trial_grouping.append(np.arange(tr_count, dtype='int64')+self.num_trials)
            self.num_trials += tr_count

        # Develop default train, validation, and test datasets 
        #self.crossval_setup() 

        ### NOTE this current will not work with multiple datasets -- just take first
        # Process touch information and store in memory
        if self.preload:
            #self.touches_full = torch.tensor(self.fhandles[f]['touches'], dtype=torch.float32)
            # Derive onset responses
            self.touches = torch.zeros([self.touches_full.shape[0], 4])

            for ww in range(4):
                Tonset = self.touches_full[1:,ww] - self.touches_full[:-1,ww]
                Tonset = torch.cat( (Tonset, torch.zeros(1)), axis=0 )
                self.touches[Tonset > 0, ww] = 1.0

            # Import RobsR and RobsL
            #self.RobsL = torch.tensor(self.fhandles[f]['RobsL'], dtype=torch.float32)
            #self.RobsR = torch.tensor(self.fhandles[f]['RobsR'], dtype=torch.float32)

            # Compute useable phase from 4 phase variables
            self.phase_ref = torch.zeros([phases.shape[0],1])
            if hemis == 1:
                wtars = [2,3]
            else:
                wtars = [0,1]

            for tt in range(self.num_trials):
                ts = self.block_inds[tt]
                if self.TRpistons[tt] >= 8:
                    self.phase_ref[ts, 0] = phases[ts, wtars[1]]
                elif self.TRpistons[tt] >= 4:
                    self.phase_ref[ts, 0] = phases[ts, wtars[0]]
                else:
                    self.phase_ref[ts, 0] = torch.mean( phases[ts, :][:, wtars], axis=1 )
        
            self.Xphase = self.create_phase_design_matrix( num_bins=num_phase_bins )

            # Process LV information initally (store in variable in memory)
            if self.LVhemis is None:
                self.LVin = None
            else:
                if (self.LVhemis == 0) or (self.LVhemis == 2):
                    LVin_tmp = torch.tensor(self.fhandles[f]['RobsL'], dtype=torch.float32)
                else:
                    LVin_tmp  = torch.tensor(self.fhandles[f]['RobsR'], dtype=torch.float32) 
                if self.hemis == 2:
                    LVin_tmp  = torch.cat(
                        (LVin_tmp,
                        torch.tensor(self.fhandles[f]['RobsR'], dtype=torch.float32)), 
                        dim=1)
                # Apply tent basis
                if LVsmooth < 2:
                    self.LVin = LVin_tmp
                else:
                    self.LVin = torch.tensor(
                        NDNutils.create_time_embedding(
                            LVin_tmp, 
                            [numLVlags, LVin_tmp.shape[1], 1], tent_spacing=self.LVsmooth),
                        dtype = torch.float32)
    
        if hemis == 0:
            self.NC = self.num_cells[0]
        elif hemis == 1:
            self.NC = self.num_cells[1]
        else:
            self.NC = self.num_cells[0] + self.num_cells[1]
    # END WhiskerData.__init__

    def crossval_setup(self, folds=5, random_gen=False, test_set=True):
        """This sets the cross-validation indices up We can add featuers here. Many ways to do this
        but will stick to some standard for now. It sets the internal indices, which can be read out
        directly or with helper functions. Perhaps helper_functions is the best way....
        
        Inputs: 
            random_gen: whether to pick random trials for validation or uniformly distributed
            test_set: whether to set aside first an n-fold test set, and then within the rest n-fold train/val sets
        Outputs:
            None: sets internal variables test_inds, train_inds, val_inds
        """

        test_fixes = []
        tfixes = []
        vfixes = []
        for ee in range(len(self.trial_grouping)):
            tr_inds = self.trial_grouping[ee]
            vfix1, tfix1 = self.fold_sample(len(tr_inds), folds, random_gen=random_gen)
            if test_set:
                test_fixes += list(tr_inds[vfix1])
                vfix2, tfix2 = self.fold_sample(len(tfix1), folds, random_gen=random_gen)
                vfixes += list(tr_inds[tfix1[vfix2]])
                tfixes += list(tr_inds[tfix1[tfix2]])
            else:
                vfixes += list(tr_inds[vfix1])
                tfixes += list(tr_inds[tfix1])

        self.val_inds = np.array(vfixes, dtype='int64')
        self.train_inds = np.array(tfixes, dtype='int64')
        if test_set:
           self.test_inds = np.array(test_fixes, dtype='int64')
    # END WhiskerData.crossval_setup

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

    def create_phase_design_matrix( self, num_bins=12 ):
        """Make design matrix of certain number of bins that maps variable of interest
        anchorL is so there is not an overall bias fit implicitly"""
        
        NT = self.phase_ref.shape[0]
        bins = np.arange(num_bins)*2*np.pi/num_bins - np.pi
        XNL = NDNutils.design_matrix_tent_basis( self.phase_ref.numpy(), bins, zero_left=True )
        return torch.tensor(XNL, dtype=torch.float32)

    ########## GET ITEM ##########
    def __getitem__(self, index):
        
        if NDNutils.is_int(index):
            index = [index]
        elif type(index) is slice:
            index = list(range(index.start or 0, index.stop or len(self.block_inds), index.step or 1))

        #stim = []
        touch, touchC = [], []
        robs = []
        dfs = []
        Xphase = []
        LVin = []
        #num_dims = self.stim_dims[0]*self.stim_dims[1]*self.stim_dims[2]
        
        for ii in index:
            inds = self.block_inds[ii]
            NT = len(inds)
            f = self.file_index[ii]
            #fix_n = self.fix_n[ii]  # which fixation, across all datasets

            """ Stim """
            # stim_tmp = torch.tensor(self.fhandles[f]['stim'][inds,:], dtype=torch.float32)
            if self.hemis == 0:
                touch_tmp = self.touches[inds,:][:,:2]
                touchC_tmp = self.touches[inds,:][:,2:]
            elif self.hemis == 1:
                touch_tmp = self.touches[inds,:][:, 2:]
                touchC_tmp = self.touches[inds,:][:, :2]
            else:
                touch_tmp = self.touches[inds,:]
                touchC_tmp = []
                #touchC_tmp = torch.zeros([len(inds),0])

            """ Xphase """
            Xphase_tmp = self.Xphase[inds, :]

            """ Robs and datafilters: needs padding so all are B x NC """ 
            if (self.hemis == 0) or (self.hemis == 2):
                if self.preload:
                    robs_tmp = self.RobsL[inds,:]
                else:
                    robs_tmp = torch.tensor(self.fhandles[f]['RobsL'][inds,:], dtype=torch.float32)
            else:
                if self.preload:
                    robs_tmp = self.RobsR[inds,:]
                else:
                    robs_tmp = torch.tensor(self.fhandles[f]['RobsR'][inds,:], dtype=torch.float32) 
            if self.hemis == 2:
                if self.preload:
                    robs_tmp = torch.cat( (robs_tmp, self.RobsR[inds, :]), dim=1 ) 
                else:
                    robs_tmp = torch.cat(
                        (robs_tmp,
                        torch.tensor(self.fhandles[f]['RobsR'][inds,:], dtype=torch.float32)), 
                        dim=1)
            
            if len(self.use_units) > 0:
                robs_tmp = robs_tmp[:, self.use_units]
            #NCbefore = int(np.asarray(self.num_units[:f]).sum())
            #NCafter = int(np.asarray(self.num_units[f+1:]).sum())
            #robs_tmp = torch.cat(
            #    (torch.zeros( (NT, NCbefore), dtype=torch.float32),
            #    robs_tmp,
            #    torch.zeros( (NT, NCafter), dtype=torch.float32)),
            #    dim=1)

            """ Datafilters: needs padding like robs """
            dfs_tmp = torch.ones(robs_tmp.shape, dtype=torch.float32)
            dfs_tmp[:self.num_lags, :] = 0.0

            #dfs_tmp[:self.num_lags,:] = 0 # invalidate the filter length
            #dfs_tmp = torch.cat(
            #    (torch.zeros( (NT, NCbefore), dtype=torch.float32),
            #    dfs_tmp,
            #    torch.zeros( (NT, NCafter), dtype=torch.float32)),
            #    dim=1)

            """ LV inputs """
            if self.LVhemis is not None:
                LVin.append(self.LVin[inds, :])

            #stim.append(stim_tmp)
            touch.append(touch_tmp)
            touchC.append(touchC_tmp)
            robs.append(robs_tmp)
            Xphase.append(Xphase_tmp)
            dfs.append(dfs_tmp)

        #stim = torch.cat(stim, dim=0)
        touch = torch.cat(touch, dim=0)
        if self.hemis < 2:
            touchC = torch.cat(touchC, dim=0)
        else:
            touchC = []

        robs = torch.cat(robs, dim=0)
        dfs = torch.cat(dfs, dim=0)
        Xphase = torch.cat(Xphase, dim=0)

        if self.LVhemis is not None:
            LVin = torch.cat(LVin, dim=0)
            return {'touch': touch, 'touchC': touchC, 'Xphase': Xphase, 'robs': robs, 'dfs': dfs, 'LVin': LVin}
        else:
            return {'touch': touch, 'touchC': touchC, 'Xphase': Xphase, 'robs': robs, 'dfs': dfs}

    ########## OTHER FUNCTIONS ##########
    def __len__(self):
        return len(self.block_inds)

    def average_firing_rates( self, valdata=True, data_inds=None):
        """Computes average firing rates for robs. valdata only uses data considering DFs
        and data_inds will look at a subset of trials"""
        if data_inds is None:
            data_inds = range(self.num_trials)
        if len(self.use_units) > 0:
            NC = len(self.use_units)
        else:
            NC = self.NC
        Rav = torch.zeros(NC)
        Tcount = torch.zeros(NC)

        for tt in data_inds:
            sample = self[tt]
            if valdata:
                Rav += torch.sum(torch.mul(sample['robs'], sample['dfs']), dim=0)
                Tcount += torch.sum(sample['dfs'], dim=0)
            else:
                Rav += torch.sum( sample['robs'], dim=0)
                Tcount += sample['robs'].shape[0] * torch.ones(NC)
        Rav = torch.div( Rav, Tcount.clamp(min=1) )
        return Rav.detach().numpy()

    def select_neurons( self, threshold=None, reset=False, valdata=True, data_inds=None ):
        if reset:
            self.use_units = []
            Ravs = self.average_firing_rates(valdata=valdata, data_inds=data_inds)
        else:
            assert threshold is not None, "Need to enter spiking-probability threshold if not reset=True."
            self.use_units = []  # goes back to full list of neurons before pulling data 
            Ravs = self.average_firing_rates(valdata=valdata, data_inds=data_inds)
            csel = np.where(Ravs >= threshold)[0]
            print("%d/%d units qualify"%(len(csel), len(Ravs)) )
            if len(csel) == 0:
                print("Threshold includes no neurons. Not set.")
            else:
                self.use_units = torch.tensor( csel, dtype=torch.int64 )
        