import tensorflow as tf
import numpy as np
import pandas as pd

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.reset_default_graph()

  my_data = [[0, 1], [2, 3], [4, 5], [6, 7]]
  slices = tf.data.Dataset.from_tensor_slices(my_data)
  next_item = slices.make_one_shot_iterator().get_next()

  writer = tf.summary.FileWriter('tf_logs')
  writer.add_graph(tf.get_default_graph())
  writer.close()

  sess= tf.Session()

  while True:
    try:
      print(sess.run(next_item))
    except tf.errors.OutOfRangeError:
      break

  sess.close()

#    while True:
#      try:
#        vv = sess.run(xx)[0]['x']
#        print(vv)
#      except tf.errors.OutOfRangeError:
#        print("End of dataset")  # ==> "End of dataset"
#        break
#
#    print("Reached end of script.")
