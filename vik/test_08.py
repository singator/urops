import tensorflow as tf
import numpy as np
import pandas as pd

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.reset_default_graph()

  a = tf.constant(3.0, dtype=tf.float32, name ='a')
  b = tf.constant(4.0, name='b') # also tf.float32 implicitly
  total = a + b

  writer = tf.summary.FileWriter('tf_logs')
  writer.add_graph(tf.get_default_graph())
  writer.close()

  with tf.Session() as sess:
    print(sess.run({'tot': total, 'ab':[a,b]}))


#    while True:
#      try:
#        vv = sess.run(xx)[0]['x']
#        print(vv)
#      except tf.errors.OutOfRangeError:
#        print("End of dataset")  # ==> "End of dataset"
#        break
#
#    print("Reached end of script.")
