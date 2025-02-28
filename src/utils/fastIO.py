
import gc
import os
import sys
import csv
import json
import codecs
import joblib
import pickle

import numpy as np
import pandas as pd
import pyarrow as arrow
import _pickle as cPickle
import pyarrow.parquet as pq

from pyarrow import csv

import warnings
warnings.filterwarnings(action='ignore')



def fastWrite_pqt(df, path, preserve_index=False):
    df.to_parquet(path, engine='pyarrow', index=preserve_index)

    
def fastWrite_pkl(data, path):
    with open(path, 'wb') as f:
        cPickle.dump(data, f, protocol=-1)


def fastRead_csv(path):
    return csv.read_csv(path).to_pandas()


def fastRead_pqt(path):
    return pd.read_parquet(path, engine='pyarrow')


def fastRead_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

