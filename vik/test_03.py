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

  dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))

  with tf.Session() as sess:
    print(dataset1.output_types)  # ==> "tf.float32"
    print(dataset1.output_shapes)  # ==> "(10,)"