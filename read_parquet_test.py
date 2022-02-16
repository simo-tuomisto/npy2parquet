import pickle
import numpy as np
from npy2parquet import iter_parquet

total_mean = 0

# Iterate over the full array
for i, (parameters, npy_array) in enumerate(iter_parquet('numpy_arrays.parquet', data_column='data', return_tuples=True), start=1):
                
        # Do calculation
        total_mean += npy_array.mean()
        
        if i%1000 == 0:
            print('Average mean of %d arrays: %f' % (i, total_mean / i))

total_mean /= i

print('Average mean of arrays: %f' % total_mean)