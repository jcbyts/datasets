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
            '20190120': 'Ellie_190120_0_0_30_30_2.mat'
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
        
        self.num_channels = 2 # vx, vy
        self.vel = vel
        Xstim = create_time_embedding( vel, [self.num_lags, self.NX*self.num_channels, self.NY], tent_spacing=1)
        self.Xstim = torch.tensor(Xstim, dtype=torch.float32)
        self.robs = torch.tensor(Robs, dtype=torch.float32)
    
    def __getitem__(self, index):
        stim = self.Xstim[index,:]
        dfs = torch.ones(stim.shape, dtype=torch.float32)
        return {'stim': stim, 'robs': self.robs[index,:], 'dfs': dfs}

    def __len__(self) -> int:
        return self.NT

    def load_set(self):
        

        Robs = self.fhandle['MoStimY'][:,:].T
        X = self.fhandle['MoStimX'][:,:].T

        Robs = Robs[:,self.cids]
        
        self.frameTime = X[:,0]

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
        self.dbin = dbin

        vel = np.concatenate((dx, dy), axis=1)
        
        # weird python reshaping
        v_reshape = np.reshape(vel,[self.NT, 2, self.NX*self.NY])
        vel = v_reshape.reshape((self.NT, self.NX*self.NY*2))

        # v_reshape = np.reshape(vel,[self.NT, 2, self.NX*self.NY])
        # vel = np.transpose(v_reshape, (0,2,1)).reshape((self.NT, self.NX*self.NY*2))

        return vel, Robs
    
    def plot_tuning_curve(self, cc, amp):

        import matplotlib.pyplot as plt
        from datasets.utils import create_time_embedding, r_squared

        amp /= np.sum(amp)
        mask = ((amp/np.max(amp)) > .5)

        X = self.fhandle['MoStimX'][:,:].T
        # stim is NT x (NX*NY). Any non-zero value is the drift direction (as an integer) of a dot (at that spatial location)
        Stim = X[:,3:]

        sfilt = Stim * mask.flatten()

        frate = 100
        inds = np.where(sfilt!=0)
        ds = sfilt[inds[0],inds[1]]
        dirs = np.unique(ds)

        dstim = np.zeros( (self.NT, len(dirs)))
        dstim[inds[0], (ds-1).astype(int)] = 1.0

        dXstim = create_time_embedding(dstim, [self.num_lags, len(dirs)])

        dsta = (dXstim.T@self.robs[:,cc].numpy()) / np.sum(dXstim, axis=0) * frate

        I = np.reshape(dsta, (-1, self.num_lags))

        dirs = dirs * self.dbin
        tpower = np.std(I,axis=0)
        peak_lag = np.argmax(tpower)

        # bootstrap error bars

        # don't sum in STA somputation (all samples preserved)
        dsta = (dXstim * np.expand_dims(self.robs[:,cc].numpy(), axis=1)) / np.sum(dXstim, axis=0) * 100

        # resample and compute confidence intervals (memory inefficient)
        nboot = 100
        bootinds = np.random.randint(0, high=self.NT, size=(self.NT, nboot))
        staboot = np.sum(dsta[bootinds,:], axis=0)
        dboot = np.reshape(staboot, (nboot, len(dirs), self.num_lags))[:,:,peak_lag]

        ci = np.percentile(dboot, (2.5, 97.5), axis=0)

        # fit von mises
        import scipy.optimize as opt

        tuning_curve = I[:,peak_lag]

        theta = np.linspace(0, 2*np.pi, 100)

        w = tuning_curve / np.sum(tuning_curve)
        th = dirs/180*np.pi
        mu0 = np.arctan2(np.sin(th)@w, np.cos(th)@w)
        bw0 = 1
        initial_guess = (mu0, bw0, np.min(tuning_curve), np.max(tuning_curve)-np.min(tuning_curve))
        popt, pcov = opt.curve_fit(von_mises, dirs/180*np.pi, tuning_curve, p0 = initial_guess)

        # plt.subplot(1,3,3)
        plt.errorbar(dirs, tuning_curve, np.abs(ci-I[:,peak_lag]), marker='o', linestyle='none', markersize=3, color='k')
        plt.plot(theta/np.pi*180, von_mises(theta, popt[0], popt[1], popt[2], popt[3]))
        plt.xlabel('Direction')
        plt.ylabel('Firing Rate (sp/s)')

        plt.xticks(np.arange(0,365,90))
        tchat = von_mises(dirs/180*np.pi, popt[0], popt[1], popt[2], popt[3])
        r2 = r_squared(np.expand_dims(tuning_curve, axis=1), np.expand_dims(tchat, axis=1))

        return {'thetas': theta/np.pi*180, 'fit': von_mises(theta, popt[0], popt[1], popt[2], popt[3]), 'dirs': dirs, 'tuning_curve': tuning_curve, 'tuning_curve_ci': np.abs(ci-I[:,peak_lag]), 'r2': r2}

    def get_rf(self, wtsAll, cc):
        
        wtsFull = wtsAll[:,cc]

        dims = [self.num_channels, self.NX, self.NY, self.num_lags]
        wts = np.reshape(wtsFull, dims)

        tpower = np.std(wts.reshape(-1,dims[-1]), axis=0)
        peak_lag = np.argmax(tpower)

        I = wts[:,:,:,peak_lag]

        dx = I[0, :,:]
        dy = I[1, :,:]

        amp = np.hypot(dx, dy)

        peak_space = np.where(amp==np.max(amp))
        min_space = np.where(amp==np.min(amp))

        ampnorm = amp / np.sum(amp)

        muw = np.array( (dx.flatten() @ ampnorm.flatten(), dy.flatten() @ ampnorm.flatten()))
        muw /= np.hypot(muw[0], muw[1])

        tpeak =  wts[0,peak_space[0],peak_space[1],:].flatten()*muw[0] + wts[1, peak_space[0],peak_space[1],:].flatten()*muw[1]
        tmin =  wts[0,min_space[0],min_space[1],:].flatten()*muw[0] + wts[1, min_space[0],min_space[1],:].flatten()*muw[1]
        lags = np.arange(0, self.num_lags, 1)*1000/100

        return {'dx': dx, 'dy': dy, 'amp': amp, 'tpeak': tpeak, 'tmin': tmin, 'lags': lags}

def von_mises(theta, thetaPref, Bandwidth, base, amplitude):

    y = base + amplitude * np.exp( Bandwidth * (np.cos(theta - thetaPref) - 1))
    return y