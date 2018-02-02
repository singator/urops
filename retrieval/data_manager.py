"""Workaround for issue 24658.
(https://bugs.python.org/issue24658)

An adaptation of:
    https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle- \
    byte-objects-larger-than-4gb
Extra functionality added for manipulating loaded data to desired format.

Docstrings to be added (010318).

Intended working directory: "."
"""

import pickle

import numpy as np
from sklearn.model_selection import train_test_split

class MacOSFile():

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size),
                # \ end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size),
                  end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size


def pickle_dump(obj, file_path):
    """Wrapper of pickle.dump"""
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_path):
    """Wrapper of pickle.load"""
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))


def create_data(lot_name, subset_size, testing_pct):
    base_path = "data/pickles/" + lot_name
    features = pickle_load(base_path + "_X.npy")
    labels = pickle_load(base_path + "_y.npy")
    
    # Even if we desire to use all the data concerning the lot, we must shuffle
    # the sets.
    if subset_size == -1:
        subset_size = features.shape[0]

    selected_indices = np.random.choice(features.shape[0],
                                        subset_size,
                                        replace=False)
    features = features[selected_indices]
    labels = labels[selected_indices]

    return train_test_split(features, labels, test_size=testing_pct)
