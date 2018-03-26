# The purpose of this script is to test how tf.data.Dataset methods work.
# The input function iris_data2.train_input_fn batches the dataset.
# 
# If you do not call repeat first, it will divide it into however many batches 
# you asked for, and return a batch every time you call it. 
# e.g.   dataset = dataset.batch(10)
# - It behaves like a generator, not an iterator. See that each batch here has
# a different number at the tail end. If it did not behave as a generator, it
# would return [0 0 0 0 0 0 0 0 0 1] every time.
# - At the end of batches, it throws an OutOfRangeError.
#
# If you repeat() it, then this error will not occur.
# e.g.   dataset = dataset.repeat().batch(10)
# 
# Remember to reload iris_data2 every time you change it.
# >>> import importlib as imp
# >>> imp.reload(iris_data2)
# 
# If you shuffle, then batch, then repeat() it, you get clear epochs, because every 
# 10 steps, it reshuffles the data and then batches it.
# e.g.   dataset = dataset.shuffle(200).batch(10).repeat()

import tensorflow as tf
import iris_data2
import numpy as np
import pandas as pd

if __name__ == '__main__':

  x = np.zeros(100, np.uint8)
  y = np.ones(100, np.uint8)
  x[np.arange(9, 100, 10, np.uint8)] = np.arange(1, 11, dtype=np.uint8)

  xd = pd.DataFrame([x1 for x1 in x],
                    columns=['x'])

  xx  = iris_data2.train_input_fn(xd, y)
  
  with tf.Session() as sess:
    for ii in np.arange(30):
      if(ii % 10 == 0):
        print('-----')
      vv = sess.run(xx)[0]['x']
      print(vv)


#    while True:
#      try:
#        vv = sess.run(xx)[0]['x']
#        print(vv)
#      except tf.errors.OutOfRangeError:
#        print("End of dataset")  # ==> "End of dataset"
#        break
#
#    print("Reached end of script.")
