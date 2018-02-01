# Not so worried about commenting and such now, just trying to create data for
# upload to Floydhub.
# Working directory: data/

import os
import pickle
import glob

import cv2
import numpy as np

# os.chdir("Documents/academic/urops/data/")

class MacOSFile(object):

    """Workaround for issue 24658."""

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
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))


names = ["solum", "primis", "secondus"]
num_spots = [100, 28, 40]
dirs = ["solum/", "../primis", "../secondus"]

for lot_index in range(3):
    os.chdir(dirs[lot_index])
    all_instances = [file for file in glob.glob("*#*")]
    num_instances = len(all_instances)
    y = np.empty(shape=(num_instances, 1), dtype="uint8")
    X = np.empty(shape=(num_instances, 32, 32, 3))

    for file_index in range(num_instances):
        fname = all_instances[file_index]
        X[file_index] = cv2.imread(fname)  # Note: BGR, not RGB.
        if fname.endswith("e.jpg"):
            y[file_index] = 0
        else:
            y[file_index] = 1

    base_path = "../pickles/" + names[lot_index]
    path_X = base_path + "_X.npy"
    path_y = base_path + "_y.npy"
    pickle_dump(X, path_X)
    pickle_dump(y, path_y)
