# Datasets
Python library to access datasets in my publications

## Datasets included
1. Mitchell Lab
  1. Free-viewing V1 Fovea
## Example

To get the 2D stimulus, V1 free-viewing dataset from the Mitchell lab, import the `pixel` dataset
```python
from datasets.mitchell.pixel import PixelDataset
```

You can get a torch `Dataset` by calling the constructor. It will look for the requested experiment in `dirname` and if it doesn't exist, it will download (as long as `download=True`)

```python
train_ds = PixelDataset('20191119',
    stimset='Train',
    stims=['Gabor', 'BackImage'],
    dirname='/home/jake/Datasets/Mitchell/stim_movies', download=True)
```

Once the stimulus is downloaded you can sample frames from it and visualize them as follows
```python
sample = train_ds[:10]
plt.figure(figsize=(10,10))
for i in range(sample['stim'].shape[0]):
    im = sample['stim'][i,0,:,:].detach().cpu().numpy()
    plt.subplot(1,sample['stim'].shape[0],i+1)
    plt.imshow(im, cmap='gray')
    plt.axis("off")
  ```
