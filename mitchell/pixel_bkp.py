
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import h5py
from tqdm import tqdm
import os
from ..utils import ensure_dir, reporthook, downsample_time

""" Available Datasets:
1. FixationMultiDataset - generates fixations from multiple stimulus classes 
                        and experimental sessions. no time-embedding
2. PixelDataset - time-embedded 2D free-viewing movies and spike trains
"""
def get_stim_list(id=None, verbose=False):

    stim_list = {
            '20191119': 'logan_20191119_-20_-10_50_60_0_19_0_1.hdf5',
            '20191120a':'logan_20191120a_-20_-10_50_60_0_19_0_1.hdf5',
            '20191121': 'logan_20191121_-20_-20_50_50_0_19_0_1.hdf5',
            '20191122': 'logan_20191122_-20_-10_50_60_0_19_0_1.hdf5',
            '20191205': 'logan_20191205_-20_-10_50_60_0_19_0_1.hdf5',
            '20191206': 'logan_20191206_-20_-10_50_60_0_19_0_1.hdf5',
            '20191231': 'logan_20191231_-20_-10_50_60_0_19_0_1.hdf5',
            '20200304': 'logan_20200304_-20_-10_50_60_0_19_0_1.hdf5'
        }

    if id is None:
        for str in list(stim_list.keys()):
            if verbose:
                print(str)
        return stim_list

    if id not in stim_list.keys():
        raise ValueError('Stimulus not found')
    
    return stim_list[id]

def get_stim_url(id):
    urlpath = {
            '20191119': 'https://www.dropbox.com/s/xxaat202j20kriy/logan_20191119_-20_-10_50_60_0_19_0_1.hdf5?dl=1',
            '20191231':'https://www.dropbox.com/s/ulpcjfb48c6dyyf/logan_20191231_-20_-10_50_60_0_19_0_1.hdf5?dl=1',
            '20200304': 'https://www.dropbox.com/s/5tj5m2rp0wht8z2/logan_20200304_-20_-10_50_60_0_19_0_1.hdf5?dl=1',
        }
    
    if id not in urlpath.keys():
        raise ValueError('Stimulus URL not found')
    
    return urlpath[id]

def get_shifter_url(id):
    urlpath = {
            '20191119': 'https://www.dropbox.com/s/dd05gxt8l8hmw3o/shifter_20191119_kilowf.p?dl=1',
            '20191120a': 'https://www.dropbox.com/s/h4elcp46le5tet0/shifter_20191120a_kilowf.p?dl=1',
            '20191121': 'https://www.dropbox.com/s/rfzefex8diu5ts5/shifter_20191121_kilowf.p?dl=1',
            '20191122': 'https://www.dropbox.com/s/2me7yauvpprnv0b/shifter_20191122_kilowf.p?dl=1',
            '20191205': 'https://www.dropbox.com/s/r56wt4rfozmjiy8/shifter_20191205_kilowf.p?dl=1',
            '20191206': 'https://www.dropbox.com/s/qec8cats077bx8c/shifter_20191206_kilowf.p?dl=1',
            '20200304': 'https://www.dropbox.com/s/t0j8k55a8jexgt4/shifter_20210304_kilowf.p?dl=1',
        }
    
    if id not in urlpath.keys():
        raise ValueError('Stimulus URL not found')
    
    return urlpath[id]


def download_set(sessname, fpath):
    
    ensure_dir(fpath)

    # Download the data set
    url = get_stim_url(sessname)
    fout = os.path.join(fpath, get_stim_list(sessname))
    print("Downloading...") 
    import urllib
    urllib.request.urlretrieve(url, fout, reporthook)
    print("Done")

def download_shifter(sessname, fpath):
    
    ensure_dir(fpath)

    # Download the data set
    url = get_shifter_url(sessname)
    fout = os.path.join(fpath, 'shifter_' + sessname + '_kilowf.p')
    print("Downloading...") 
    import urllib
    urllib.request.urlretrieve(url, fout, reporthook)
    print("Done")


def shift_im(im, shift):
        """
        apply shifter to translate stimulus as a function of the eye position
        im = N x C x H x W (torch.float32)
        shift = N x 2 (torch.float32)
        """
        import torch.nn.functional as F
        import torch
        affine_trans = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])
        sz = im.shape

        aff = torch.tensor([[1,0,0],[0,1,0]])

        affine_trans = shift[:,:,None]+aff[None,:,:]
        affine_trans[:,0,0] = 1
        affine_trans[:,0,1] = 0
        affine_trans[:,1,0] = 0
        affine_trans[:,1,1] = 1

        n = im.shape[0]
        grid = F.affine_grid(affine_trans, torch.Size((n, 1, sz[2], sz[3])), align_corners=False)

        im2 = F.grid_sample(im, grid, align_corners=False)

        return im2.detach()

class FixationMultiDataset(Dataset):

    def __init__(self,
        sess_list,
        dirname,
        stimset="Train",
        requested_stims=["Gabor"],
        downsample_s: int=1,
        downsample_t: int=2,
        num_lags: int=12,
        num_lags_pre_sac: int=12,
        saccade_basis = None,
        max_fix_length: int=1000,
        download=True,
        flatten=True,
        crop_inds=None,
        min_fix_length: int=50,
        valid_eye_rad=5.2,
        add_noise=0,
        verbose=True):

        self.dirname = dirname
        self.stimset = stimset
        self.requested_stims = requested_stims
        self.downsample_s = downsample_s
        self.downsample_t = downsample_t
        self.spike_sorting = 'kilowf' # only one option for now
        self.valid_eye_rad = valid_eye_rad
        self.min_fix_length = min_fix_length
        self.flatten = flatten
        self.num_lags = num_lags
        self.num_lags_pre_sac = num_lags_pre_sac
        self.normalizing_constant = 50
        self.max_fix_length = max_fix_length
        self.saccade_basis = saccade_basis
        self.shift = None # default shift to None. To provide shifts, set outside this class. Should be a list of shift values equal to size dataset.eyepos in every way
        self.add_noise = add_noise

        if self.saccade_basis is not None:
            if type(self.saccade_basis) is np.array:
                self.saccadeB = self.saccade_basis
            else:
                if type(self.saccade_basis) is not dict or 'max_len' not in self.saccade_basis.keys():
                    self.saccade_basis['max_len'] = 40
                if type(self.saccade_basis) is not dict or 'num' not in self.saccade_basis.keys():
                    self.saccade_basis['num'] = 15
                self.saccadeB = np.maximum(1 - np.abs(np.expand_dims(np.asarray(np.arange(0,self.saccade_basis['max_len'])), axis=1) - np.arange(0,self.saccade_basis['max_len'],self.saccade_basis['max_len']/self.saccade_basis['num']))/self.saccade_basis['max_len']*self.saccade_basis['num'], 0)
        else:
            self.saccadeB = None
            
        # find valid sessions
        stim_list = get_stim_list() # list of valid sessions
        new_sess = []
        for sess in sess_list:
            if sess in stim_list.keys():
                if verbose:
                    print("Found [%s]" %sess)
                new_sess.append(sess)

        self.sess_list = new_sess # is a list of valid sessions
        self.fnames = [get_stim_list(sess) for sess in self.sess_list] # is a list of filenames

        # check if files exist. download if they don't
        for isess,fname in enumerate(self.fnames):
            fpath = os.path.join(dirname, fname)
            if not os.path.exists(fpath):
                print("File [%s] does not exist. Download set to [%s]" % (fpath, download))
                if download:
                    print("Downloading set...")
                    download_set(self.sess_list[isess], dirname)
                else:
                    print("Download is False. Exiting...")

        # open hdf5 files as a list of handles
        self.fhandles = [h5py.File(os.path.join(dirname, fname), 'r') for fname in self.fnames]

        # build index map
        self.file_index = [] # which file the fixation corresponds to
        self.stim_index = [] # which stimulus the fixation corresponds to
        self.fixation_inds = [] # the actual index into the hdf5 file for this fixation
        self.eyepos = []

        self.unit_ids_orig = []
        self.unit_id_map = []
        self.unit_ids = []
        self.num_units = []
        self.NC = 0       
        self.time_start = 0
        self.time_stop = 0

        for f, fhandle in enumerate(self.fhandles): # loop over experimental sessions
            
            # store neuron ids
            # unique_units = np.unique(fhandle['Neurons'][self.spike_sorting]['cluster'][:]).astype(int)
            unique_units = fhandle['Neurons'][self.spike_sorting]['cids'][0,:].astype(int)
            self.unit_ids_orig.append(unique_units)
            
            # map unit ids to index into the new ids
            mc = np.max(unique_units)+1
            unit_map = -1*np.ones(mc, dtype=int)
            unit_map[unique_units] = np.arange(len(unique_units))+self.NC
            self.unit_id_map.append(unit_map)

            # number of units in this session
            self.num_units.append(len(self.unit_ids_orig[-1]))

            # new unit ids
            self.unit_ids.append(np.arange(self.num_units[-1])+self.NC)
            self.NC += self.num_units[-1]

            # loop over stimuli
            for s, stim in enumerate(self.requested_stims): # loop over requested stimuli
                if stim in fhandle.keys(): # if the stimuli exist in this session
                    
                    sz = fhandle[stim]['Train']['Stim'].attrs['size']
                    self.dims = [1, int(sz[0]), int(sz[1])]

                    # get fixationss
                    labels = fhandle[stim][stimset]['labels'][:] # labeled eye positions
                    labels = labels.flatten()
                    labels[0] = 0 # force ends to be 0, so equal number onsets and offsets
                    labels[-1] = 0
                    fixations = np.diff( (labels ==  1).astype(int)) # 1 is fixation
                    fixstart = np.where(fixations==1)[0]
                    fixstop = np.where(fixations==-1)[0]

                    # offset to include lags before fixation onset
                    fixstart = fixstart-self.num_lags_pre_sac
                    fixstop = fixstop[fixstart>=0]
                    fixstart = fixstart[fixstart>=0]

                    nfix = len(fixstart)
                    if verbose:
                        print("%d fixations" %nfix)

                    # get valid indices
                    # get blocks (start, stop) of valid samples
                    blocks = fhandle[stim][stimset]['blocks'][:,:]
                    valid_inds = []
                    for bb in range(blocks.shape[1]):
                        valid_inds.append(np.arange(blocks[0,bb],
                        blocks[1,bb]))
        
                    valid_inds = np.concatenate(valid_inds).astype(int)

                    for fix_ii in range(nfix): # loop over fixations
                        
                        # get the index into the hdf5 file
                        fix_inds = np.arange(fixstart[fix_ii]+1, fixstop[fix_ii])
                        fix_inds = np.intersect1d(fix_inds, valid_inds)
                        if len(np.where(np.diff(fhandle[stim][stimset]['frameTimesOe'][0,fix_inds])>0.01)[0]) > 1:
                            if verbose:
                                print("dropped frames. skipping %d" %fix_ii)

                        if len(fix_inds) > self.max_fix_length:
                            fix_inds = fix_inds[:self.max_fix_length]

                        # check if the fixation meets our requirements to include
                        # sample eye pos
                        ppd = fhandle[stim][self.stimset]['Stim'].attrs['ppd'][0]
                        centerpix = fhandle[stim][self.stimset]['Stim'].attrs['center'][:]
                        eye_tmp = fhandle[stim][self.stimset]['eyeAtFrame'][1:3,fix_inds].T
                        eye_tmp[:,0] -= centerpix[0]
                        eye_tmp[:,1] -= centerpix[1]
                        eye_tmp/= ppd

                        if len(fix_inds) < self.min_fix_length:
                            if verbose:
                                print("fixation too short. skipping %d" %fix_ii)
                            continue
                        
                        # is the eye position outside the valid region?
                        if np.mean(np.hypot(eye_tmp[(self.num_lags_pre_sac+5):,0], eye_tmp[(self.num_lags_pre_sac+5):,1])) > self.valid_eye_rad:
                            if verbose:
                                print("eye outside valid region. skipping %d" %fix_ii)
                            continue

                        dx = np.diff(eye_tmp, axis=0)
                        vel = np.hypot(dx[:,0], dx[:,1])
                        vel[:self.num_lags_pre_sac+5] = 0
                        # find missed saccades
                        potential_saccades = np.where(vel[5:]>0.1)[0]
                        if len(potential_saccades)>0:
                            sacc_start = potential_saccades[0]
                            valid = np.arange(0, sacc_start)
                        else:
                            valid = np.arange(0, len(fix_inds))
                        
                        if len(valid)>self.min_fix_length:
                            self.eyepos.append(eye_tmp[valid,:])
                            self.fixation_inds.append(fix_inds[valid])
                            self.file_index.append(f) # which datafile does the fixation correspond to
                            self.stim_index.append(s) # which stimulus does the fixation correspond to
            
            if crop_inds is None:
                self.crop_inds = [0, self.dims[1], 0, self.dims[2]]
            else:
                self.crop_inds = [crop_inds[0], crop_inds[1], crop_inds[2], crop_inds[3]]
                self.dims[1] = crop_inds[1]-crop_inds[0]
                self.dims[2] = crop_inds[3]-crop_inds[2]

    def __getitem__(self, index):
        """
        Get item for a Fixation dataset.
        Each element in the index corresponds to a fixation. Each fixation will have a variable length.
        Concatenate fixations along the batch dimension.

        """
        stim = []
        robs = []
        dfs = []
        eyepos = []
        frames = []
        sacB = []
        fix_n = []
        

        # handle indices (can be a range, list, int, or slice). We need to convert ints, and slices into an iterable for looping
        if isinstance(index, int) or isinstance(index, np.int64):
            index = [index]
        elif isinstance(index, slice):
            index = list(range(index.start or 0, index.stop or len(self.fixation_inds), index.step or 1))
        # loop over fixations
        for ifix in index:
            fix_inds = self.fixation_inds[ifix] # indices into file for this fixation
            file = self.file_index[ifix]
            stimix = self.stim_index[ifix] # stimulus index for this fixation

            """ STIMULUS """
            # THIS is the only line that matters for sampling the stimulus if your file is already set up right
            I = self.fhandles[file][self.requested_stims[stimix]][self.stimset]['Stim'][:,:,fix_inds]
            
            # bring the individual values into a more reasonable range (instead of [-127,127])
            I = I.astype(np.float32)/self.normalizing_constant
            
            I = torch.tensor(I, dtype=torch.float32).permute(2,0,1) # [H,W,N] -> [N,H,W]

            if self.add_noise>0:
                I += torch.randn(I.shape)*self.add_noise
            
            # append the stimulus to the list of tensors
            if self.shift is not None:
                I = shift_im(I.unsqueeze(1), self.shift[ifix])
            else:
                I = I.unsqueeze(1)
            
            I = I[...,self.crop_inds[0]:self.crop_inds[1],self.crop_inds[2]:self.crop_inds[3]]

            stim.append(I)
            fix_n.append(torch.ones(I.shape[0], dtype=torch.int64)*ifix)

            """ SPIKES """
            # NOTE: normally this would look just like the line above, but for 'Robs', but I am operating with spike times here
            # NOTE: this is MUCH slower than just indexing into the file
            frame_times = self.fhandles[file][self.requested_stims[stimix]][self.stimset]['frameTimesOe'][0,fix_inds]
            frame_times = np.expand_dims(frame_times, axis=1)
            if self.downsample_t>1:
                frame_times = downsample_time(frame_times, self.downsample_t, flipped=False)

            frames.append(torch.tensor(frame_times, dtype=torch.float32))
            frame_times = frame_times.flatten()

            spike_inds = np.where(np.logical_and(
                self.fhandles[file]['Neurons'][self.spike_sorting]['times']>=frame_times[0],
                self.fhandles[file]['Neurons'][self.spike_sorting]['times']<=frame_times[-1]+0.01)
                )[1]

            st = self.fhandles[file]['Neurons'][self.spike_sorting]['times'][0,spike_inds]
            clu = self.fhandles[file]['Neurons'][self.spike_sorting]['cluster'][0,spike_inds].astype(int)
            # only keep spikes that are in the requested cluster ids list
            ix = np.in1d(clu, self.unit_ids_orig[file])
            st = st[ix]
            clu = clu[ix]
            # map cluster id to a unit number
            clu = self.unit_id_map[file][clu]
            
            # do the actual binning
            # robs_tmp = bin_population(st, clu, frame_times, self.unit_ids[file], maxbsize=1.2/240)
            
            # if self.downsample_t>1:
            #     robs_tmp = downsample_time(robs_tmp, self.downsample_t, flipped=False)

            robs_tmp = torch.sparse_coo_tensor( np.asarray([np.digitize(st, frame_times)-1, clu]),
                 np.ones(len(clu)), (len(frame_times), self.NC) , dtype=torch.float32)
            robs_tmp = robs_tmp.to_dense()
            robs.append(robs_tmp)

            """ DATAFILTERS """
            NCbefore = int(np.asarray(self.num_units[:file]).sum())
            NCafter = int(np.asarray(self.num_units[file+1:]).sum())
            dfs_tmp = torch.cat(
                (torch.zeros( (len(frame_times), NCbefore), dtype=torch.float32),
                torch.ones( (len(frame_times), self.num_units[file]), dtype=torch.float32),
                torch.zeros( (len(frame_times), NCafter), dtype=torch.float32)),
                dim=1)
            dfs_tmp[:self.num_lags,:] = 0 # temporal convolution will be invalid for the filter length

            # if self.downsample_t>1:
            #     dfs_tmp = downsample_time(dfs_tmp, self.downsample_t, flipped=False)

            dfs.append(dfs_tmp)

            """ EYE POSITION """
            ppd = self.fhandles[file][self.requested_stims[stimix]][self.stimset]['Stim'].attrs['ppd'][0]
            centerpix = self.fhandles[file][self.requested_stims[stimix]][self.stimset]['Stim'].attrs['center'][:]
            eye_tmp = self.fhandles[file][self.requested_stims[stimix]][self.stimset]['eyeAtFrame'][1:3,fix_inds].T
            eye_tmp[:,0] -= centerpix[0]
            eye_tmp[:,1] -= centerpix[1]
            eye_tmp/= ppd

            assert np.all(self.eyepos[ifix] == eye_tmp), 'eyepos does not match between object and file'
            
            if self.downsample_t>1:
                eye_tmp = downsample_time(eye_tmp, self.downsample_t, flipped=False)

            eyepos.append(torch.tensor(eye_tmp, dtype=torch.float32))

            """ SACCADES (on basis) """
            if self.saccadeB is not None:
                fix_len = len(frame_times)
                sacB_len = self.saccadeB.shape[0]
                if fix_len < sacB_len:
                    sacc_tmp = torch.tensor(self.saccadeB[:fix_len,:], dtype=torch.float32)
                else:
                    sacc_tmp = torch.cat( (torch.tensor(self.saccadeB, dtype=torch.float32),
                        torch.zeros( (fix_len-sacB_len, self.saccadeB.shape[1]), dtype=torch.float32)
                        ), dim=0)
                sacB.append(sacc_tmp)

        # concatenate along batch dimension
        stim = torch.cat(stim, dim=0)
        fix_n = torch.cat(fix_n, dim=0)
        eyepos = torch.cat(eyepos, dim=0)
        robs = torch.cat(robs, dim=0)
        dfs = torch.cat(dfs, dim=0)
        frames = torch.cat(frames, dim=0)

        if self.flatten:
            stim = torch.flatten(stim, start_dim=1)
        
        sample = {'stim': stim, 'robs': robs, 'dfs': dfs, 'eyepos': eyepos, 'frame_times': frames, 'fix_n': fix_n}

        if self.saccadeB is not None:
            sample['saccade'] = torch.cat(sacB, dim=0)

        return sample

    def __len__(self):
        return len(self.fixation_inds)

    def get_stim_indices(self, stim_name='Gabor'):
        if isinstance(stim_name, str):
            stim_name = [stim_name]
        stim_id = [i for i,s in enumerate(self.requested_stims) if s in stim_name]

        indices = [i for i,s in enumerate(self.stim_index) if s in stim_id]
        
        return indices
    
    def expand_shift(self, fix_shift, fix_inds=None):
        if fix_inds is None:
            assert fix_shift.shape[0]==len(self), 'fix_shift not equal number of fixations. Pass in fix_inds as well.'
            fix_inds = np.arange(len(self)) # index into all fixations
        
        new_shift = []
        for i, fix in enumerate(fix_inds):
            new_shift.append(fix_shift[i].repeat(len(self.fixation_inds[fix]), 1))
        
        new_shift = torch.cat(new_shift, dim=0)
        return new_shift

    def plot_shifter(self, shifter, valid_eye_rad=5.2, ngrid = 100):
        import matplotlib.pyplot as plt
        xx,yy = np.meshgrid(np.linspace(-valid_eye_rad, valid_eye_rad,ngrid),np.linspace(-valid_eye_rad, valid_eye_rad,ngrid))
        xgrid = torch.tensor( xx.astype('float32').reshape( (-1,1)))
        ygrid = torch.tensor( yy.astype('float32').reshape( (-1,1)))

        inputs = torch.cat( (xgrid,ygrid), dim=1)

        xyshift = shifter(inputs).detach().numpy()

        xyshift/=valid_eye_rad/60 # conver to arcmin
        vmin = np.min(xyshift)
        vmax = np.max(xyshift)

        shift = [xyshift[:,0].reshape((ngrid,ngrid))]
        shift.append(xyshift[:,1].reshape((ngrid,ngrid))) 
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        plt.imshow(shift[0], extent=(-valid_eye_rad,valid_eye_rad,-valid_eye_rad,valid_eye_rad), interpolation=None, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.subplot(1,2,2)
        plt.imshow(shift[1], extent=(-valid_eye_rad,valid_eye_rad,-valid_eye_rad,valid_eye_rad), interpolation=None, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.show()

        return shift

    def get_shifters(self, plot=False):

        shifters = {}
        for sess in self.sess_list:
            sfname = [f for f in os.listdir(self.dirname) if 'shifter_' + sess in f]
                
            if len(sfname) == 0:
                from datasets.mitchell.pixel import download_shifter
                download_shifter(self.sess_list[0], self.dirname)
            else:
                print("Shifter exists")
                import pickle
                fname = os.path.join(self.dirname, sfname[0])
                shifter_res = pickle.load(open(fname, "rb"))
                shifter = shifter_res['shifters'][np.argmin(shifter_res['vallos'])]

            if plot:
                _ = self.plot_shifter(shifter)
            
            shifters[sess] = shifter

        return shifters
    

    def shift_stim(self, im, shift, unflatten=False):
        """
        apply shifter to translate stimulus as a function of the eye position
        """
        import torch.nn.functional as F
        import torch
        affine_trans = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])
        sz = [im.shape[0]] + self.dims

        if len(im.shape)==2:
            unflatten = True
            im = im.reshape(sz)

        aff = torch.tensor([[1,0,0],[0,1,0]])

        affine_trans = shift[:,:,None]+aff[None,:,:]
        affine_trans[:,0,0] = 1
        affine_trans[:,0,1] = 0
        affine_trans[:,1,0] = 0
        affine_trans[:,1,1] = 1

        n = im.shape[0]
        grid = F.affine_grid(affine_trans, torch.Size((n, 1, sz[-2], sz[-1])), align_corners=False)

        im2 = F.grid_sample(im, grid, align_corners=False)

        if unflatten:
            torch.flatten(im2, start_dim=1)

        return im2


class Pixel(Dataset):
    
    def __init__(self,
        dirname,
        sess_list:list=None,
        stimset:str="Train",
        requested_stims:list=["Gabor"],
        num_lags:int=12,
        cids:list=None,
        flatten:bool=False,
        dim_order:str='cwht',
        fixations_only:bool=False,
        valid_eye_rad:float=5.2,
        downsample_t:int=1,
        verbose=False,
        download=False,
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
                    
                    self.stim_indices[expt][stim] = {'inds': np.arange(runninglength, runninglength + sz[-1])}
                    self.valid_idx.append(runninglength + self.get_valid_indices(self.fhandles[expt], stim))

                    runninglength += sz[-1]
                    
                    self.dims[1] = np.maximum(self.dims[1], sz[0])
                    self.dims[2] = np.maximum(self.dims[1], sz[1])
        
        self.valid_idx = np.concatenate(self.valid_idx)
        
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
                    sz = fhandle[stim][stimset]['Stim'].shape
                    inds = self.stim_indices[expt][stim]['inds']

                    self.stim[0, :sz[0], :sz[1], inds] = np.transpose(fhandle[stim][stimset]['Stim'][...], [2,0,1])
                    self.frame_times[inds] = fhandle[stim][stimset]['frameTimesOe'][...].T


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
                    self.robs[inds,:] = robs_tmp

                    """ DATAFILTERS """
                    unit_ids = self.spike_indices['20200304']['unit ids']
                    for unit in unit_ids:
                        self.dfs[inds, unit] = 1

                    """ EYE POSITION """
                    ppd = fhandle[stim][self.stimset]['Stim'].attrs['ppd'][0]
                    centerpix = fhandle[stim][self.stimset]['Stim'].attrs['center'][:]
                    eye_tmp = fhandle[stim][self.stimset]['eyeAtFrame'][1:3,:].T
                    eye_tmp[:,0] -= centerpix[0]
                    eye_tmp[:,1] -= centerpix[1]
                    eye_tmp/= ppd
                    self.eyepos[inds,:] = eye_tmp


        
        



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

    def __len__(self):
        return len(self.valid_idx)
    
    def __getitem__(self, idx):
                    
        r = self.robs[self.valid_idx[idx],:]
        s = self.stim[..., self.valid_idx[idx,None]-range(self.num_lags)]
        if len(s.shape)==5:
            s = s.transpose(3,0,1,2,4)
        
        return {'stim': torch.Tensor(s.astype('float32')), 'robs': torch.Tensor(r.astype('float32'))}
    
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



class PixelDataset(Dataset):
    """
    PixelDataset is a pytorch Dataset for loading stimulus movies and spikes
    Arguments:
    id:             <string>    id of the session (must exist in get_stim_list, e.g,. '20200304)
    num_lags:       <int>       number of time lags
    stims:          <list>      list of strings corresponding to requested stimuli
                                    "Gabor"     - gabor droplet noise
                                    "Grating"   - full-field gratings
                                    "BackImage" - static natural image 
                                    "FixRsvpStim" - rapidly flashed filtered natural images
    stimset:        <string>    "Train" or "Test" set
    downsample_s    <int>       spatial downsample factor (this slows things down, because smoothing before subsample)
    downsample_t    <int>       temporal downsample factor (this slows things down because of smoothing operation)
    valid_eye_rad   <float>     valid region on screen (centered on 0,0) in degrees of visual angle
    fixations_only  <bool>      whether to only include fixations
    dirname         <string>    full path to where the data are stored
    cids            <list>      list of cells to include (file is loaded in its entirety and then sampled because slicing hdf5 has to be simple)
    cropidx                     index of the form [(x0,x1),(y0,y1)] or None type
    include_eyepos  <bool>      flag to include the eye position info in __get_item__ output
    temporal        <bool>      include temporal dimension explicitly instead of buried as channel dimension

    """
    def __init__(self,id,
        num_lags:int=1,
        stimset="Train",
        stims=["Gabor"],
        downsample_s: int=1,
        downsample_t: int=2,
        valid_eye_rad=5.2,
        valid_eye_ctr=(0.0,0.0),
        fixations_only=True,
        dirname='/home/jake/Data/Datasets/MitchellV1FreeViewing/stim_movies/',
        download=False,
        cids=None,
        cropidx=None,
        shifter=None,
        preload=False,
        include_eyepos=True,
        dim_order='chwt',
        include_saccades=None, # must be a dict that says how to implement the saccade basis
        include_frametime=None,
        optics=None,
        temporal=True,
        flatten=True,
        device=torch.device('cpu'),
        dtype=torch.float32):
        
        
        self.device = device
        self.dtype = dtype

        if optics is None:
            optics = {'type': 'none', 'sigma': (0,0,0)}

        # check if a specific spike sorting is requested
        chk = [i for i,j in zip(range(len(id)), id) if '_'==j ]

        if len(chk)==0:
            sessname = id
            spike_sorting = None
        else:
            sessname = id[:chk[0]]
            spike_sorting = id[chk[0]+1:]
        
        if flatten: # flatten overrides temporal (which keeps a unit channel dimension)
            temporal = False

        # load data
        self.dirname = dirname
        self.id = id
        self.cropidx = cropidx
        self.fname = get_stim_list(sessname)
        self.sdnorm = 15 # scale stimuli (puts model in better range??)
        self.stimset = stimset
        self.flatten = flatten
        self.dim_order = dim_order

        # check if we need to download the data
        fpath = os.path.join(self.dirname, self.fname)
        if not os.path.exists(fpath):
            print("File [%s] does not exist. Download set to [%s]" % (fpath, download))
            if download:
                print("Downloading set...")
                download_set(sessname, self.dirname)
            else:
                print("Download is False. Exiting...")
                return

        self.fhandle = h5py.File(os.path.join(self.dirname,self.fname), "r")
        self.isopen = True
        self.num_lags = num_lags
        self.downsample_s = downsample_s
        self.downsample_t = downsample_t
        self.fixations_only = fixations_only
        self.include_eyepos = include_eyepos
        self.include_saccades = include_saccades
        self.include_frametime = include_frametime
        self.valid_eye_rad = valid_eye_rad
        self.valid_eye_ctr = valid_eye_ctr
        self.shifter=shifter
        self.temporal = temporal
        self.spike_sorting = spike_sorting
        self.optics = optics

        # sanity check stimuli (all requested stimuli must be keys in the file)
        newstims = []
        for s in range(len(stims)):
            if stims[s] in self.fhandle.keys():
                newstims.append(stims[s])
        print("Found requested stimuli %s" %newstims)
        self.stims = newstims

        # useful info to pull from meta data
        sz = self.fhandle[self.stims[0]]['Train']['Stim'].attrs['size']
        ppd = self.fhandle[self.stims[0]]['Train']['Stim'].attrs['ppd'][0]
        self.centerpix = self.fhandle[self.stims[0]]['Train']['Stim'].attrs['center'][:]
        self.rect = self.fhandle[self.stims[0]]['Train']['Stim'].attrs['rect'][:]
        self.ppd = ppd
        self.NY = int(sz[0]//self.downsample_s)
        self.NX = int(sz[1]//self.downsample_s)
        self.NF = 1 # number of channels
        self.frate = self.fhandle[self.stims[0]]['Test']['Stim'].attrs['frate'][0]

        # get valid indices
        self.valid = [self.get_valid_indices(stim) for stim in self.stims]
        self.lens = [len(v) for v in self.valid]
        indices = [[i] * v for i, v in enumerate(self.lens)]
        self.stim_indices = np.asarray(sum(indices, []))
        self.indices_hack = np.arange(0,len(self.stim_indices)) # stupid conversion from slice/int/range to numpy array

        # pre-load frame times        
        self.frame_time = np.zeros(len(self.stim_indices))
        
        for ii,stim in zip(range(len(self.stims)), self.stims):
            inds = self.stim_indices==ii
            self.frame_time[inds] = self.fhandle[stim][self.stimset]['frameTimesOe'][0,self.valid[ii]]
        """
        LOAD FRAME TIMES AS TENT BASIS
        We want to represent time in the experiment as a smoothly varying parameter so we can fit up-and-down states, excitability, artifacts, etc.
        """
        if self.include_frametime is not None:
            print("Loading frame times on basis")
            assert type(self.include_frametime)==dict, "include_frametime must be a dict with keys: 'num_basis', 'full_experiment'"
            
            assert not self.include_frametime['full_experiment'], "full_experiment = True is not implemented yet"
            self.frame_basiscenters = np.linspace(np.min(self.frame_time), np.max(self.frame_time), self.include_frametime['num_basis'])
            self.frame_binsize = np.mean(np.diff(self.frame_basiscenters))
            xdiff = np.abs(np.expand_dims(self.frame_time, axis=1) - self.frame_basiscenters)
            self.frame_tents = np.maximum(1-xdiff/self.frame_binsize , 0)
        
        """
        INCLUDE SACCADES
        Build design matrix for saccade lags
        
        include_saccades is a list of dicts. Each dict has arguments for how to construct the basis
        {
            'name': name of the variable
            'basis': actual basis (at the time resolution of the stimulus)
        }
        """
        if self.include_saccades is not None:
            assert type(self.include_saccades)==list, "include_saccades must be a list of dicts with keys: 'name', 'offset', 'basis'"
            from scipy.signal import fftconvolve
            print("Loading saccade times on basis")
            self.saccade_times = []
            for isacfeature in range(len(self.include_saccades)):
                self.saccade_times.append( np.zeros( (len(self.stim_indices), self.include_saccades[isacfeature]['basis'].shape[1])))

            for ii,stim in zip(range(len(self.stims)), self.stims):
                print("%d) %s" %(ii,stim))
                labels = self.fhandle[stim][self.stimset]['labels'][0,:]
                sactimes = np.diff((labels==2).astype('float32'))
                for isacfeature in range(len(self.include_saccades)):
                    if self.include_saccades[isacfeature]['name']=="sacon":
                        sacstim = (sactimes==1).astype('float32')
                    elif self.include_saccades[isacfeature]['name']=="sacoff":
                        sacstim = (sactimes==-1).astype('float32')

                    # shift forward or backward to make acasual or delayed lags
                    off = self.include_saccades[isacfeature]['offset']
                    sacstim = np.roll(sacstim, off)
                    # zero out invalid after shift
                    if off < 0:
                        sacstim[off:] = 0
                    elif off > 0:
                        sacstim[:off] = 0
                    
                    # convolve with basis
                    sacfull = fftconvolve(np.expand_dims(sacstim, axis=1), self.include_saccades[isacfeature]['basis'], axes=0)

                    # index into valid times
                    inds = self.stim_indices==ii
                    self.saccade_times[isacfeature][inds,:] = sacfull[self.valid[ii],:]



        # setup cropping
        if cropidx:
            self.cropidx = cropidx
            self.NX = cropidx[0][1] - cropidx[0][0]
            self.NY = cropidx[1][1] - cropidx[1][0]
        else:
            self.cropidx = None

        # spike meta data / specify clusters
        if self.spike_sorting is not None:
            self.cluster_ids = self.fhandle['Neurons'][self.spike_sorting]['cids'][0,:]
            cgs = self.fhandle['Neurons'][self.spike_sorting]['cgs'][0,:]
        else:
            self.cluster_ids = self.fhandle[self.stims[0]]['Test']['Robs'].attrs['cids']
            if 'cgs' in self.fhandle['Neurons'].keys():
                cgs = self.fhandle['Neurons']['cgs'][:][0]
            else:
                cgs = np.ones(len(self.cluster_ids))*2
        
        self.NC = len(self.cluster_ids)
        if cids is not None:
            self.cids = cids
            self.NC = len(cids)
            self.cluster_ids = self.cluster_ids[cids]
        else:
            self.cids = list(range(0,self.NC-1))
        
        # self.single_unit = [int(cgs[c])==2 for c in self.cids]

        if self.spike_sorting is not None:
            from V1FreeViewingCode.Analysis.notebooks.Utils import bin_at_frames
            """
            HANDLE specific spike sorting
            """
            st = self.fhandle['Neurons'][spike_sorting]['times'][0,:]
            clu = self.fhandle['Neurons'][spike_sorting]['cluster'][0,:]
            cids = self.cluster_ids

            Robs = np.zeros((len(self.frame_time), self.NC))
            inds = np.argsort(self.frame_time)
            ft = self.frame_time[inds]
            for cc in range(self.NC):
                cnt = bin_at_frames(st[clu==cids[cc]], ft, maxbsize=1.2/self.frate)
                Robs[inds,cc] = cnt
            self.y = torch.tensor(Robs.astype('float32'), device=self.device, dtype=self.dtype)
        
        if preload: # preload data if it will fit in memory
            self.preload=False
            print("Preload True. Loading ")
            n = len(self)
            if self.flatten:
                self.x = torch.ones((n,self.NF*self.NY*self.NX*self.num_lags), device=self.device, dtype=self.dtype)
            else:
                if self.dim_order=='cxyt':
                    self.x = torch.ones((n,self.NF, self.NY, self.NX, self.num_lags), device=self.device, dtype=self.dtype)
                elif self.dim_order=='txy':
                    self.x = torch.ones((n,self.num_lags, self.NY, self.NX), device=self.device, dtype=self.dtype)

            if self.spike_sorting is None:
                self.y = torch.ones((n,self.NC), device=self.device, dtype=self.dtype)
            self.eyepos = torch.ones( (n,2), device=self.device, dtype=self.dtype)
            chunk_size = 10000
            nsteps = n//chunk_size+1
            for i in range(nsteps):
                print("%d/%d" %(i+1,nsteps))
                inds = np.arange(i*chunk_size, np.minimum(i*chunk_size + chunk_size, n))
                sample = self.__getitem__(inds)
                self.x[inds,:] = sample['stim'] #.detach().clone()
                if self.spike_sorting is None:
                    self.y[inds,:] = sample['robs'] # .detach().clone()
                self.eyepos[inds,0] = sample['eyepos'][:,0]#.detach().clone()
                self.eyepos[inds,1] = sample['eyepos'][:,1]#.detach().clone()
            print("Done")
            self.datafilters = torch.ones(self.y.shape, device=self.device, dtype=self.dtype)
            self.preload = True
        self.preload = preload

    def __getitem__(self, index):
        """
            This is a required Dataset method
        """            
        
        if self.preload:
            if self.temporal:
                if type(index)==int:
                    stim = self.x[index,:].unsqueeze(0)
                else:
                    stim = self.x[index,:].unsqueeze(1) # TODO: check that implicit dimensions are handled correctly
            else:
                stim = self.x[index,:]
            
            out = {'stim': stim, 'robs': self.y[index,:], 'eyepos': self.eyepos[index,:], 'dfs': self.datafilters[index,:]}

            if self.include_frametime is not None:
                out['frametime'] = torch.tensor(self.frame_tents[index,:].astype('float32'))
                out['frame_times'] = torch.Tensor(self.frame_time[index,None].astype('float32'))
            
            if self.include_saccades is not None:
                for ii in range(len(self.saccade_times)):
                    out[self.include_saccades[ii]['name']] = torch.tensor(self.saccade_times[ii][index,:].astype('float32'))

            return out
        else:  
            if self.optics['type']=='gausspsf':
                from scipy.ndimage import gaussian_filter

            if type(index)==int: # special case where a single instance is indexed
                inisint = True
            else:
                inisint = False

            # index into valid stimulus indices (this is part of handling multiple stimulus sets)
            uinds, uinverse = np.unique(self.stim_indices[index], return_inverse=True)
            indices = self.indices_hack[index] # this is now a numpy array

            
            # loop over stimuli included in this index
            for ss in range(len(uinds)):
                istim = uinds[ss] # index into stimulus
                stim_start = np.where(self.stim_indices==istim)[0][0]
                # stim_inds = np.where(uinverse==ss)[0] - stim_start
                stim_inds = indices - stim_start
                ix = uinverse==ss
                if inisint:
                    valid_inds = self.valid[istim][stim_inds]
                    file_inds = valid_inds - range(0,self.num_lags*self.downsample_t)
                else:
                    stim_inds = stim_inds[ix]
                    valid_inds = self.valid[istim][stim_inds]
                    file_inds = np.expand_dims(valid_inds, axis=1) - range(0,self.num_lags*self.downsample_t)
                    
                ufinds, ufinverse = np.unique(file_inds.flatten(), return_inverse=True)
                if self.cropidx and not self.shifter:
                    I = self.fhandle[self.stims[istim]][self.stimset]["Stim"][self.cropidx[1][0]:self.cropidx[1][1],self.cropidx[0][0]:self.cropidx[0][1],ufinds]
                else:
                    I = self.fhandle[self.stims[istim]][self.stimset]["Stim"][:,:,ufinds]

                if self.shifter:
                    eyepos = self.fhandle[self.stims[istim]][self.stimset]["eyeAtFrame"][1:3,ufinds].T
                    eyepos[:,0] -= self.centerpix[0]
                    eyepos[:,1] -= self.centerpix[1]
                    eyepos/= self.ppd
                    I = self.shift_stim(I, eyepos)
                    if self.cropidx:
                        I = I[self.cropidx[1][0]:self.cropidx[1][1],self.cropidx[0][0]:self.cropidx[0][1],:]
                
                if self.optics['type']=='gausspsf':
                    I = gaussian_filter(I, self.optics['sigma'])

                if self.spike_sorting is None:
                    R = self.fhandle[self.stims[istim]][self.stimset]["Robs"][:,valid_inds]
                    R = R.T

                if self.include_eyepos:
                    eyepos = self.fhandle[self.stims[istim]][self.stimset]["eyeAtFrame"][1:3,valid_inds].T
                    if inisint:
                        eyepos[0] -= self.centerpix[0]
                        eyepos[1] -= self.centerpix[1]
                    else:    
                        eyepos[:,0] -= self.centerpix[0]
                        eyepos[:,1] -= self.centerpix[1]
                    eyepos/= self.ppd

                sz = I.shape
                
                if inisint:
                    I = np.expand_dims(I, axis=3)

                # expand time dimension to native H x W x B x T
                I = I[:,:,ufinverse].reshape(sz[0],sz[1],-1, self.num_lags*self.downsample_t)

                # move batch dimension into 0th spot and re-order the remaining dims
                if 'hwt' in self.dim_order:
                    I = I.transpose((2,0,1,3))
                elif 'thw' in self.dim_order:
                    I = I.transpose((2,3,0,1))
                else:
                    raise ValueError('PixelDataset: Unknown dim_order, must be thw or hwt')

                if self.spike_sorting is None:
                    if inisint:
                        NumC = len(R)
                    else:
                        NumC = R.shape[1]

                    if self.NC != NumC:
                        if inisint:
                            R = R[np.asarray(self.cids)]
                        else:
                            R = R[:,np.asarray(self.cids)]

                # concatentate if necessary
                if ss ==0:
                    S = torch.tensor(self.transform_stim(I), device=self.device, dtype=self.dtype)
                    if self.spike_sorting is None:
                        Robs = torch.tensor(R, device=self.device, dtype=self.dtype)

                    if self.include_eyepos:
                        ep = torch.tensor(eyepos, device=self.device, dtype=self.dtype)
                    else:
                        ep = None

                    # if inisint:
                    #     S = S[0,:,:,:] #.unsqueeze(0)
                    #     Robs = Robs[0,:] #.unsqueeze(0)
                    #     ep = ep[0,:] #.unsqueeze(0)

                else:
                    S = torch.cat( (S, torch.tensor(self.transform_stim(I), device=self.device, dtype=self.dtype)), dim=0)
                    if self.spike_sorting is None:
                        Robs = torch.cat( (Robs, torch.tensor(R, device=self.device, dtype=self.dtype)), dim=0)
                    if self.include_eyepos:
                        ep = torch.cat( (ep, torch.tensor(eyepos, device=self.device, dtype=self.dtype)), dim=0)

            if self.temporal:
                if inisint:
                    S = S.unsqueeze(0) # add channel dimension
                else:
                    S = S.unsqueeze(1) # add channel dimension

            if self.spike_sorting is not None:
                Robs = self.y[index,:]

            if self.flatten:
                if inisint:
                    S = torch.flatten(S, start_dim=0)
                else:
                    S = torch.flatten(S, start_dim=1)

            out = {'stim': S, 'robs': Robs, 'eyepos': ep, 'dfs': torch.ones(Robs.shape, device=self.device, dtype=self.dtype)}

            if self.include_frametime is not None:
                out['frametime'] = torch.tensor(self.frame_tents[index,:].astype('float32'))
                out['frame_times'] = torch.Tensor(self.frame_time[index,None].astype('float32'))
            
            if self.include_saccades is not None:
                for ii in range(len(self.saccade_times)):
                    out[self.include_saccades[ii]['name']] = torch.tensor(self.saccade_times[ii][index,:].astype('float32'))

            return out

    def __len__(self):
        return sum(self.lens)

    def get_valid_indices(self, stim):
        # get blocks (start, stop) of valid samples
        blocks = self.fhandle[stim][self.stimset]['blocks'][:,:]
        valid = []
        for bb in range(blocks.shape[1]):
            valid.append(np.arange(blocks[0,bb]+self.num_lags*self.downsample_t,
                blocks[1,bb])) # offset start by num_lags
        
        valid = np.concatenate(valid).astype(int)

        if self.fixations_only:
            fixations = np.where(self.fhandle[stim][self.stimset]['labels'][:]==1)[1]
            valid = np.intersect1d(valid, fixations)
        
        if self.valid_eye_rad:
            xy = self.fhandle[stim][self.stimset]['eyeAtFrame'][1:3,:].T
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

    def transform_stim(self, s):
        # stim comes in N,Lags,Y,X
        s = s.astype('float32')/self.sdnorm

        if self.downsample_t>1 or self.downsample_s>1:
            from scipy.ndimage import gaussian_filter
            
            if 'hwt' in self.dim_order:
                sig = [0, self.downsample_s-1, self.downsample_s-1, self.downsample_t-1] # smoothing before downsample
                s = gaussian_filter(s, sig)
                s = s[:,::self.downsample_s,::self.downsample_s,::self.downsample_t]
            elif 'thw' in self.dim_order:
                sig = [0, self.downsample_t-1, self.downsample_s-1, self.downsample_s-1] # smoothing before downsample
                s = gaussian_filter(s, sig)
                s = s[:,::self.downsample_t,::self.downsample_s,::self.downsample_s]
            else:
                raise ValueError('PixelDataset: Unknown dim_order, must be thw or hwt')

        if 'c' == self.dim_order[0]:
            s = np.expand_dims(s, axis=1)

        if s.shape[0]==1:
            s=s[0,:,:,:] # return single item

        return s


    def shift_stim(self, im, eyepos):
        """
        apply shifter to translate stimulus as a function of the eye position
        """
        import torch.nn.functional as F
        import torch
        affine_trans = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])
        sz = im.shape
        eyepos = torch.tensor(eyepos.astype('float32'))
        im = torch.tensor(im[:,None,:,:].astype('float32'))
        im = im.permute((3,1,0,2))

        shift = self.shifter(eyepos).detach()
        aff = torch.tensor([[1,0,0],[0,1,0]])

        affine_trans = shift[:,:,None]+aff[None,:,:]
        affine_trans[:,0,0] = 1
        affine_trans[:,0,1] = 0
        affine_trans[:,1,0] = 0
        affine_trans[:,1,1] = 1

        n = im.shape[0]
        grid = F.affine_grid(affine_trans, torch.Size((n, 1, sz[0], sz[1])), align_corners=True)

        im2 = F.grid_sample(im, grid, align_corners=True)
        im2 = im2[:,0,:,:].permute((1,2,0)).detach().cpu().numpy()

        return im2
    
    def get_null_adjusted_ll_temporal(self,model,batch_size=5000):
        """
        Get raw and null log-likelihood for all time points
        """
        import torch
        loss = torch.nn.PoissonNLLLoss(log_input=False, reduction='none')
        

        nt = len(self)
        llneuron = np.zeros((nt, self.NC))
        llnull = np.zeros((nt, self.NC))
        robs = np.zeros((nt,self.NC))
        robshat = np.zeros((nt,self.NC))
        eyepos = np.zeros((nt,2))

        nsteps = nt//batch_size + 1

        model.cpu()

        print("Getting log-likelihood for all time bins")
        for istep in tqdm(range(nsteps)):

            if nsteps==1:   
                index = np.arange(0,nt)
            else:
                index = (istep-1)*batch_size + np.arange(0, batch_size)
                index = index[index < nt]
                
            sample = self[index]
            try:
                yhat = model(sample['stim'], shifter=sample['eyepos'], sample=sample)
            except TypeError:
                yhat = model(sample['stim'], shifter=sample['eyepos'])

            llneuron[index,:] = -loss(yhat,sample['robs']).detach().cpu().numpy()
            llnull[index,:] = -loss(torch.ones(sample['robs'].shape)*sample['robs'].mean(axis=0), sample['robs']).detach().cpu().numpy()
            robs[index,:] = sample['robs']
            robshat[index,:] = yhat.detach()
            eyepos[index,:] = sample['eyepos']
        
        return {'llraw': llneuron, 'llnull': llnull, 'robshat': robshat, 'robs': robs, 'eyepos': eyepos}

    def get_ll_by_eyepos(self, model, lldict=None, nbins=20, binsize=.5, bounds=[-5,5],
        batch_size=5000, plot=True, use_stim=None):
        """
        get_ll_by_eyepos gets the null-adjusted loglikelihood for each cell as a function of eye position
        """
        if lldict is None:
            lldict = self.get_null_adjusted_ll_temporal(model, batch_size=batch_size)

        import numpy as np
        import matplotlib.pyplot as plt
        print("Getting log-likelihood as a function of eye position")

        bins = np.linspace(bounds[0],bounds[1],nbins)

        LLspace = np.zeros((nbins,nbins,self.NC))

        if plot:
            sx = np.ceil(np.sqrt(self.NC))
            sy = np.round(np.sqrt(self.NC))
            plt.figure(figsize=(3*sx,3*sy))

        for cc in tqdm(range(self.NC)):
            for ii,xx in zip(range(nbins),bins):
                for jj,yy in zip(range(nbins),bins):
                    ix = np.hypot(lldict['eyepos'][:,0] - xx, lldict['eyepos'][:,1]-yy) < binsize
                    if not use_stim is None:
                        ix = np.logical_and(self.stim_indices==use_stim, ix)
                    LLspace[jj,ii,cc] = np.mean(lldict['llraw'][ix,cc]-lldict['llnull'][ix,cc])

            if plot:
                plt.subplot(sx,sy,cc+1)
                plt.imshow(LLspace[:,:,cc], extent=[bounds[0],bounds[1],bounds[0],bounds[1]])
                plt.title(cc)

        return LLspace
    
    def get_null_adjusted_ll(self, model, sample=None, bits=False, use_shifter=True):
        '''
        get null-adjusted log likelihood
        bits=True will return in units of bits/spike
        '''
        m0 = model.cpu()
        loss = torch.nn.PoissonNLLLoss(log_input=False, reduction='none')
        if sample is None:
            sample = self[:]

        lnull = -loss(torch.ones(sample['robs'].shape)*sample['robs'].mean(axis=0), sample['robs']).detach().cpu().numpy().sum(axis=0)
        if use_shifter:
            yhat = m0(sample['stim'], shifter=sample['eyepos'])
        else:
            yhat = m0(sample['stim'], sample=sample)
        llneuron = -loss(yhat,sample['robs']).detach().cpu().numpy().sum(axis=0)
        rbar = sample['robs'].sum(axis=0).numpy()
        ll = (llneuron - lnull)/rbar
        if bits:
            ll/=np.log(2)
        return ll
                # plt.colorbar()