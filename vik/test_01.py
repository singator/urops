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

# Our application logic will be added here

if __name__ == "__main__":

#  with open('../data/pX.npy', 'rb') as f:
#    pX = pickle.load(f)
#  with open('../data/pY.npy', 'rb') as f:
#    pY = pickle.load(f)
#
#  pX = pX[0:20]
#  pY = pY[0:20]
#
#  assert pX.shape[0] == pY.shape[0]
#  
#  dataset1 = tf.data.Dataset.from_tensor_slices(({'r':pX}, {'y':pY}))
#  iterator = dataset1.batch(2).make_one_shot_iterator()
#  n1,n2 = iterator.get_next()

  dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 2]))
  dataset2 = tf.data.Dataset.from_tensor_slices((tf.random_uniform([4]), 
    tf.random_uniform([4, 8])))
  dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
  iterator = dataset3.make_initializable_iterator()
  next_element = iterator.get_next()

  with tf.Session() as sess:
    sess.run(iterator.initializer)
    while True:
      try:
        print(sess.run(next_element))
      except tf.errors.OutOfRangeError:
        print("Exiting now!")
        break
