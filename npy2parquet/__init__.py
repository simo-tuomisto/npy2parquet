# -*- coding: utf-8 -*-
"""This module converts numpy arrays listed in a dataframe into a parquet-file.
"""

from npy2parquet.functions import df2parquet, npy2bytes, bytes2ndarray, iter_parquet, verify_parquet
