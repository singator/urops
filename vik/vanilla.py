from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import pickle
import argparse
from sklearn.model_selection import train_test_split

tf.logging.set_verbosity(tf.logging.INFO)
parser = argparse.ArgumentParser()

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

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(({"pixels": features}, labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

def main(argv):
  args = parser.parse_args(argv[1:])

  pX = pickle_load('../data/primis_X.npy')
  pY = pickle_load('../data/primis_Y.npy').ravel()

  train_X, test_X, train_Y, test_Y = train_test_split(pX, pY, test_size=0.2,
      random_state=2002)
  val_X, test_X, val_Y, test_Y = train_test_split(test_X, test_Y,
      test_size=0.5, random_state=2003)

# stratification counts:
  print('{:.2%} of the'.format((np.bincount(train_Y)/len(train_Y))[0]), 
      'training set is "occupied"')
  print('{:.2%} of the'.format((np.bincount(val_Y)/len(val_Y))[0]), 
    'validation set is "occupied"')
  print('{:.2%} of the'.format((np.bincount(test_Y)/len(test_Y))[0]), 
      'test set is "occupied"')

# Try:
  classifier = tf.estimator.DNNClassifier(
      feature_columns=[tf.feature_column.numeric_column(key="pixels")],
      # Two hidden layers of 10 nodes each.
      hidden_units=[10, 10],
      # The model must choose between 3 classes.
      n_classes=2,
      model_dir=".")

  classifier.train(
      input_fn=lambda:train_input_fn(train_X, train_Y, 512), steps=100)

# Our application logic will be added here

if __name__ == "__main__":
  tf.app.run(main)
