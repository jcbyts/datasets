import os
import sys
import time

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