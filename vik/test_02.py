from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

if __name__ == "__main__":

  tf.reset_default_graph()

  k = tf.placeholder(tf.float32, name="multiplier_for_mean")

  # Make a normal distribution, with a shifting mean
  mean_moving_normal = tf.random_normal(shape=[1000], mean=5.0*k, stddev=1.0)
  # Record that distribution into a histogram summary
  x = tf.summary.histogram("normal/moving_mean", mean_moving_normal)
  merged=tf.summary.merge_all()

  # Setup a loop and write the summaries to disk
  N = 400

  with tf.Session() as sess:

    writer = tf.summary.FileWriter("tmp/histogram_example_finally", sess.graph)

    for ii in range(N):

      k_val = ii/float(N)
      summ = sess.run(merged, feed_dict={k: k_val})
      writer.add_summary(summ, ii)

    writer.close()
