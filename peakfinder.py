"""
A GPU compatible clone of photutils peakfinder

https://photutils.readthedocs.io/en/stable/_modules/photutils/detection/peakfinder.html
"""
import torch
import torch.nn.functional as F

def find_peaks(data, threshold, box_size=3, np_ar=None):
    """
    Box Size needs to be odd for downscale and upscale to work for now...

    parameters
    ----------

    outputs
    -------
    """
    def maximum_filter(x, kernel_size):
        pad = (kernel_size - 1)//2
        x = F.max_pool2d(x, kernel_size, stride=1, padding=pad)
        return x

    data_max = maximum_filter(data, box_size)

    peak_goodmask = (data == data_max)

    peak_goodmask = torch.logical_and(peak_goodmask, (data > threshold))

    locations = peak_goodmask.nonzero(as_tuple=True)
    peak_values = data[locations]

    return locations, peak_values

if __name__ == "__main__":
    from photutils.detection import find_peaks as p_fp
    import numpy as np
    from astropy.io import fits
    import glob
    ar = fits.open("/data/astroml/jgeach/DeepConfusion/testing_images_fixed_model/august_true_N07180.00_S02.50_gamma1.50_noise3.56_0007.fits")[0].data
    ar = torch.from_numpy(ar.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    par = p_fp(ar.squeeze(), box_size=4, threshold=1)
    tar, vals = find_peaks(ar, threshold=1, np_ar=ar.squeeze().numpy())
    par = np.asarray(par["peak_value"])
    vals = vals.numpy()
    print(len(set(par) - set(vals)))
    print(len(set(vals) - set(par)))
    print(len(set(vals) & set(par)))

    #print(tar)
