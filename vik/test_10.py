import pickle
import os
import time
import argparse

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# The following hyperparameters are global variables because `cnn_model_fn()`
# cannot accept any arguments except those currently listed, in order to
# conform with the `Estimator` API.
# Both of the below are set when the function is called from the command line--
# i.e. they are flags.
global dropout_rate, learning_rate


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


def pickle_load(file_path):
  """Wrapper of pickle.load"""
  with open(file_path, "rb") as f:
      return pickle.load(MacOSFile(f))

if __name__ == "__main__":
  tmp = pickle_load("../data/" + "split_primis.npy")
  train_X, test_X, train_y, test_y = tmp[0], tmp[1], tmp[2], tmp[3]

  print(len(train_X))
