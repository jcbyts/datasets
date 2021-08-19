import os
import sys
import time
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import torch

def download_file(url: str, file_name: str):
    '''
    Downloads file from url and saves it to file_name.
    '''
    import urllib.request
    print("Downloading %s to %s" % (url, file_name))
    urllib.request.urlretrieve(url, file_name, reporthook)
    
def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

def ensure_dir(dir_name: str):
    '''
    Creates folder if not exists.
    '''
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def downsample_time(x, ds, flipped=None):
    
    NTold = x.shape[0]
    dims = x.shape[1]
    
    if flipped is None:
        flipped = False
        if dims > NTold:
	        # then assume flipped
	        flipped = True
	        x = x.T
    
    NTnew = np.floor(NTold/ds).astype(int)
    if type(x) is torch.Tensor:
        y = torch.zeros((NTnew, dims), dtype=x.dtype)
    else:
        y = np.zeros((NTnew, dims))
        
    for nn in range(ds-1):
        y[:,:] = y[:,:] + x[nn + np.arange(0, NTnew, 1)*ds,:]
    
    if flipped:
        y = y.T
        
    return y

def resample_time(data, torig, tout):
    ''' resample data at times torig at times tout '''
    ''' data is components x time '''
    ''' credit to Carsen Stringer '''
    fs = torig.size / tout.size # relative sampling rate
    data = gaussian_filter1d(data, np.ceil(fs/4), axis=1)
    f = interp1d(torig, data, kind='linear', axis=0, fill_value='extrapolate')
    dout = f(tout)
    return dout

def bin_population(times, clu, btimes, cids,
        maxbsize=1, padding=0, dtype=torch.float32):
    ''' bin time points (times) at btimes'''
    NC = np.max(cids) + 1
    robs = torch.zeros((len(btimes), NC), dtype=dtype)
    inds = np.argsort(btimes)
    ft = btimes[inds]
    for cc in range(NC):
        cnt = bin_at_frames(times[clu==cids[cc]], ft, maxbsize=maxbsize, padding=padding)
        robs[inds,cc] = torch.tensor(cnt, dtype=dtype)
    return robs

def bin_population_sparse(times, clu, btimes, cids, dtype=torch.float32, to_dense=True):
    ''' bin time points (times) at btimes'''
    NC = np.max(cids)+1
    robs = torch.sparse_coo_tensor( (np.digitize(times, btimes)-1, clu), np.ones(len(clu)), (len(btimes), NC) , dtype=dtype)
    if to_dense:
        return robs.to_dense()
    else:
        return robs

def bin_at_frames(times, btimes, maxbsize=1, padding=0):
    ''' bin time points (times) at btimes'''
    breaks = np.where(np.diff(btimes)>maxbsize)[0]
    
    # add extra bin edge
    btimes = np.append(btimes, btimes[-1]+maxbsize)

    out,_ = np.histogram(times, bins=btimes)
    out = out.astype('float32')

    if padding > 0:
        out2 = out[range(breaks[0])]
        dt = np.median(np.diff(btimes))
        pad = np.arange(1,padding+1, 1)*dt
        for i in range(1,len(breaks)):
            tmp,_ = np.histogram(times, pad+btimes[breaks[i]])
            out2.append(tmp)
            out2.append(out[range(breaks[i-1]+1, breaks[i])])            
    else:
        out[breaks] = 0.0
    
    return out

def r_squared(true, pred, data_indxs=None):
    """
    START.

    :param true: vector containing true values
    :param pred: vector containing predicted (modeled) values
    :param data_indxs: obv.
    :return: R^2

    It is assumed that vectors are organized in columns

    END.
    """

    assert true.shape == pred.shape, 'true and prediction vectors should have the same shape'

    if data_indxs is None:
        dim = true.shape[0]
        data_indxs = np.arange(dim)
    else:
        dim = len(data_indxs)

    ss_res = np.sum(np.square(true[data_indxs, :] - pred[data_indxs, :]), axis=0) / dim
    ss_tot = np.var(true[data_indxs, :], axis=0)

    return 1 - ss_res/ss_tot