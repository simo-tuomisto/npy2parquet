# -*- coding: utf-8 -*-
"""Functions that can convert numpy arrays listed in a dataframe into a parquet-file.
"""

import os
import pickle
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def npy2bytes(npy_file):
    """Helper function for loading bytes from .npy-file.

    This helper function returns the contents of a .npy-file as bytes.

    Parameters
    ----------
    npy_file : str
        Name of a .npy file.

    Returns
    -------
    bytes
        Data in numpy array as bytes.
    """
    return np.load(npy_file).dumps()


def bytes2ndarray(data):
    """Helper function for loading numpy.ndarray from dumped pickle.

    This helper function returns a unpickled numpy.ndarray.

    Parameters
    ----------
    data : bytes
        Pickled numpy.ndarray.

    Returns
    -------
    numpy.ndarray
        Unpickled numpy.ndarray.
    """
    return pickle.loads(data)


def df2parquet(files_dataframe,
               parquet_file,
               npyfile_column,
               data_column,
               batch_size=50,
               shuffle=False,
               seed=10,
               overwrite=False):
    """This function converts a dataframe that contains a column of numpy array
    names into a parquet-file with numpy array data included.

    Given a pandas DataFrame, where one column npyfile_column houses a list of
    npy-filenames, this function will write a parquet-file where the data in
    the npy-files will be stored into a new column data_column.

    Parameters
    ----------
    files_dataframe : :obj:`pandas.DataFrame`
        DataFrame that will be used for the creation of the parquet-file.
    parquet_file : str
        Name of the output .parquet-file.
    npyfile_column : str
        Name of the column that has the numpy files.
    data_column : str
        Name of the column that will store the numpy data in the parquet-file.
    batch_size : :obj:`int`, optional
        Number of files to store within one batch inside of the parquet-file.
        Warning: The amount of data within one batch needs to fit into computer
        memory.
    shuffle : :obj:`bool`, optional
        Whether row ordering should be shuffled or not before writing.
    seed : :obj:`int`, optional
        Seed for shuffling.
    overwrite : :obj:`bool`, optional
        Whether output file should be overwritten if it exists.

    Raises
    ------
    RuntimeError
        Returns RuntimeError if file already exists and overwrite is False.
    """

    if os.path.isfile(parquet_file):
        if overwrite:
            print(f"File {parquet_file} exists, removing it.")
            os.remove(parquet_file)
        else:
            raise RuntimeError("File exists, will not overwrite as 'overwrite' is False")

    pqwriter = None

    if shuffle:
        print(f'Shuffling labels with seed {seed}.')
        rng = np.random.default_rng(seed=seed)
        index_shuffle = np.asarray(files_dataframe.index)
        rng.shuffle(index_shuffle)
        files_dataframe.index = index_shuffle
        files_dataframe.sort_index(inplace=True)

    nsplits = int(np.ceil(float(len(files_dataframe))/batch_size))
    print(f'Splitting the dataframe with {len(files_dataframe)} rows '
          f'to {nsplits} batches of {batch_size}.')
    batch_split = np.array_split(files_dataframe, nsplits)

    for index, df_chunk in enumerate(batch_split):
        df_chunk[data_column] = df_chunk[npyfile_column].apply(npy2bytes)

        pa_table = pa.Table.from_pandas(df_chunk, preserve_index=True)
        if index == 0:
            pqwriter = pq.ParquetWriter(parquet_file, pa_table.schema)
        pqwriter.write_table(pa_table, row_group_size=batch_size)

    pqwriter.close()


def iter_parquet(parquet_file,
                 data_column,
                 batch_size=50,
                 shuffle=False,
                 seed=10,
                 return_tuples=False):
    """This function iterates a parquet-file that contains a column of numpy
    arrays.

    Given a parquet-file, where one column has pickled numpy arrays, this
    function will iterate throught the array in sequential or pseudo-random
    order.

    Parameters
    ----------
    parquet_file : str
        Parquet-file, that will be iterated over.
    data_column : str
        Name of the column that stores the numpy data in the parquet-file.
    shuffle : :obj:`bool`, optional
        Whether row ordering should be shuffled or during reading.
    batch_size : :obj:`int`, optional
        Number of files to store within one batch inside of the parquet-file.
        Warning: The amount of data within one batch needs to fit into computer
        memory.
    seed : :obj:`int`, optional
        Seed for shuffling.
    return_tuples : bool
        Return named tuples instead of :obj:`pandas.DataFrame` as the first
        return value.

    Yields
    -------
    pandas.DataFrame
        Other columns in the row of the parquet-file
    numpy.ndarray
        Unpickled ndarray

    """

    # Open parquet-file
    pa_table = pq.read_table(parquet_file, use_pandas_metadata=True)

    # Split the file into batches
    batches = pa_table.to_batches(max_chunksize=batch_size)

    # Shuffle batch ordering
    if shuffle:
        print(f'Shuffling labels with seed {seed}.')
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(batches)

    # For each batch, read the data
    for batch in batches:

        # Read data from batch into memory as a dataframe
        batch_df = batch.to_pandas()

        # Shuffle rows in each batch
        if shuffle:
            index_shuffle = np.asarray(batch_df.index)
            rng.shuffle(index_shuffle)
            batch_df.index = index_shuffle
            batch_df.sort_index(inplace=True)

        # Pop data column into its own series so that we can iterate over it separately
        data_series = batch_df.pop(data_column)

        # Iterate over rest of the dataframe and over the data series
        for batch_tuple, data in zip(batch_df.itertuples(), data_series):

            if not return_tuples:
                batch_tuple = pd.DataFrame([batch_tuple]).set_index('Index', drop=True)
                batch_tuple.index.name = batch_df.index.name

            # Convert pickled ndarray into a reguler ndarray
            npy_array = bytes2ndarray(data)

            # Return other columns in the dataframe and the ndarray
            yield batch_tuple, npy_array


def verify_parquet(parquet_file,
                   files_dataframe,
                   npyfile_column,
                   data_column,
                   batch_size=50,
                   shuffle=False,
                   seed=10):
    """This function verifies that the data stored in a parquet-file matches
    the numpy arrays that are specified in a files_dataframe.


    Given a parquet-file, where one column has pickled numpy arrays, and
    a pandas.Dataframe, where one column has the names of the numpy arrays,
    this function will verify that the data matches. Warning: It will iterate
    over the full dataset.


    Parameters
    ----------
    parquet_file : str
        Parquet-file, that will be iterated over.
    files_dataframe : :obj:`pandas.DataFrame`
        DataFrame that will be used for the verification of the parquet-file.
    npyfile_column : str
        Name of the column that has the numpy files in files_dataframe.
    data_column : str
        Name of the column stores the numpy data in the parquet-file.
    batch_size : :obj:`int`, optional
        Number of files to store within one batch inside of the parquet-file.
        Warning: The amount of data within one batch needs to fit into computer
        memory.
    shuffle : :obj:`bool`, optional
        Whether row ordering should be shuffled or not during reading.
    seed : :obj:`int`, optional
        Seed for shuffling.

    Returns
    -------
    bool
        Returns True, if all columns in the files_dataframe and
        parquet_file match. False otherwise.
    bool
        Returns True, if all data in the .npy files match the data in
        the .parquet-file. False otherwise.

    """

    parquet_iterator = iter_parquet(parquet_file,
                                    data_column=data_column,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    seed=seed)

    df_rows_equal = []
    data_equal = []

    # Store old index name for later on
    orig_index_name = files_dataframe.index.name

    # Use npyfile_column as an index for easier searching
    files_dataframe = files_dataframe.reset_index() \
                                     .set_index(npyfile_column, drop=False)

    for parquet_df_row, parquet_data in parquet_iterator:

        # Test dataframes

        # Pick npyfile-column for searching the original dataframe
        search_column = parquet_df_row[npyfile_column]

        # Choose the correct row from the dataframe
        files_dataframe_df_row = files_dataframe.loc[search_column, :]

        # Remove sorting index and reset index to it's original state
        files_dataframe_df_row = files_dataframe_df_row.reset_index(drop=True)
        files_dataframe_df_row.set_index('index', inplace=True)
        files_dataframe_df_row.index.name = orig_index_name

        # Record if data & columns in dataframes are equal
        df_rows_equal.append(files_dataframe_df_row.equals(parquet_df_row))

        # Test numpy data

        # Find the name of the numpyfile
        numpyfile = parquet_df_row[npyfile_column].iloc[0]

        # Load data from file
        npy_data = np.load(numpyfile)

        # Test data in numpy arrays
        data_equal.append(np.all(npy_data == parquet_data))

    n_rows = files_dataframe.shape[0]
    n_row_equal = np.sum(df_rows_equal)
    n_data_equal = np.sum(data_equal)

    print(f'Ran tests for {parquet_file}:')
    print('')
    print('DataFrame tests:')
    print(f'Number of rows:                     {n_rows}')
    print(f'Number of matching rows:            {n_row_equal}')
    print(f'Number of mismatching rows:         {n_rows - n_row_equal}')
    print('')
    print('Numpy arrays tests:')
    print(f'Number of arrays:                   {n_rows}')
    print(f'Number of matching numpy arrays:    {n_data_equal}')
    print(f'Number of mismatching numpy arrays: {n_rows - n_data_equal}')

    return np.all(df_rows_equal), np.all(data_equal)
