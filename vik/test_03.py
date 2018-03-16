from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

"""
This is finally working and yielding the correct graphs.
I have learnt a lot about tensorflow today. 

We can use tf.app(main) or keep everything within __main__
It is important to reset the graph. The clue was in:
  'Delete the underlying status object from memory...'
"""


if __name__ == "__main__":

  # This is such an important line. The documentation for tensorflow is very
  # poor.
  tf.reset_default_graph()

  x = tf.placeholder(tf.float32, shape=(), name="x123234")
  y = tf.Variable(initial_value=0.123, dtype=tf.float32, name="Y1232")
  y = 2*x
  tf.summary.scalar("x_s", x)
  tf.summary.scalar("y_s", y)
  merged = tf.summary.merge_all()

  with tf.Session() as sess:

    writer = tf.summary.FileWriter("tmp/log124", sess.graph)

    for ii in range(200):
      summ = sess.run(merged, feed_dict={x: float(ii)})
      writer.add_summary(summ, ii)

    writer.close()
