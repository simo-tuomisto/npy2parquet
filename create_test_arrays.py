import os
import argparse
from glob import glob

import numpy as np
import pandas as pd


def create_test_arrays(narrays=100, shape=(1000,1000)):
    
    print('Creating %d arrays of shape %s' % (narrays, shape)) 

    array_folder = 'numpy_arrays'
    array_format = os.path.join('numpy_arrays', 'array-%s.npy')

    # Create folder for arrays

    if not os.path.isdir(array_folder):
        os.makedirs(array_folder)

    # Remove old numpy arrays

    for array in glob(array_format % '*'):
        os.remove(array)

    # Create numpy arrays

    arrays_data = []
    for n in list(range(narrays)):
            arr = np.random.random(shape)
            array_name = array_format % n
            np.save(array_name, arr)
            # Random values that could represent some parameters
            random_parameters = np.random.random(2)
            arrays_data.append([n,random_parameters[0],random_parameters[1],array_name])

    # Store parameters and array filenames in a CSV file

    df = pd.DataFrame(columns=('n','param1','param2','npyfile'), data=arrays_data)

    df.to_csv('numpy_arrays.csv', index=False)

    print(df)
    
    print('Numpy arrays are described in numpy_arrays.csv')
    
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(usage='This program will write random numpy arrays to numpy_arrays')
    parser.add_argument('-x', '--x-size', default=1000, type=int, help='Size of first dimension in arrays')
    parser.add_argument('-y', '--y-size', default=1000, type=int, help='Size of second dimension in arrays')
    parser.add_argument('-n', '--narrays', default=100, type=int, help='Number of arrays')
    
    args = parser.parse_args()
        
    create_test_arrays(narrays=args.narrays, shape=(args.x_size, args.y_size))