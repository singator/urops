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

def train_input_fn(features, labels):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(({'x':features}, labels))

    # Shuffle, repeat, and batch the examples.
    # dataset = dataset.shuffle(100).repeat().batch(batch_size)

    # Return the dataset.
    return dataset.batch(2)

def main(argv):
  args = parser.parse_args(argv[1:])

  with open('../data/pX.npy', 'rb') as f:
    pX = pickle.load(f)
  with open('../data/pY.npy', 'rb') as f:
    pY = pickle.load(f)

  pX = pX[0:20]
  pY = pY[0:20]

  assert pX.shape[0] == pY.shape[0]

#  train_X, test_X, train_Y, test_Y = train_test_split(pX, pY, test_size=0.2,
#      random_state=2002)
#  val_X, test_X, val_Y, test_Y = train_test_split(test_X, test_Y,
#      test_size=0.5, random_state=2003)

# stratification counts:
#  print('{:.2%} of the'.format((np.bincount(train_Y)/len(train_Y))[0]), 
#      'training set is "occupied"')
#  print('{:.2%} of the'.format((np.bincount(val_Y)/len(val_Y))[0]), 
#    'validation set is "occupied"')
#  print('{:.2%} of the'.format((np.bincount(test_Y)/len(test_Y))[0]), 
#      'test set is "occupied"')

# Try:
  x1 = tf.feature_column.numeric_column(key='x')
  classifier = tf.estimator.DNNClassifier(
      feature_columns=[x1],
      # Two hidden layers of 10 nodes each.
      hidden_units=[3, 3],
      # The model must choose between 3 classes.
      n_classes=2,
      model_dir=".")

  classifier.train(
      input_fn=lambda:train_input_fn(pX, pY), steps=1)

# Our application logic will be added here

if __name__ == "__main__":
  tf.app.run(main)
