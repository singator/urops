""" Create and pickle feature and label matrices for the three lots under
current consideration: primis, secondus, and solum.

Intended working directory: ./data/
 """

import os
import glob

import cv2
import numpy as np

# os.chdir("/Users/nurmister/Documents/academic/urops/retrieval/")

import data_manager

# os.chdir("/Users/nurmister/Documents/academic/urops/data/")

names = ["solum", "primis", "secondus"]
num_spots = [100, 28, 40]
dirs = ["solum/", "../primis", "../secondus"]

for lot_index in range(3):
    os.chdir(dirs[lot_index])
    all_instances = [file for file in glob.glob("*#*")]
    num_instances = len(all_instances)
    y = np.empty(shape=(num_instances, 1), dtype="int32")
    X = np.empty(shape=(num_instances, 32, 32, 3), dtype="float32")

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
    data_manager.pickle_dump(X, path_X)
    data_manager.pickle_dump(y, path_y)
