# Iterate simply with numpy arrays

import numpy as np

import glob

total_mean = 0

for i, npy_datafile in enumerate(glob.glob('numpy_arrays/*.npy')):
    npy_array = np.load(npy_datafile)

    # Do calculation
    total_mean += npy_array.mean()

    i += 1
    if i%1000 == 0:
        print('Average mean of %d arrays: %f' % (i, total_mean / i))


total_mean /= i

print('Average mean of arrays: %f' % total_mean)