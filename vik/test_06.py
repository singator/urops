#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""An Example of a DNNClassifier for the Iris dataset."""
import tensorflow as tf
import iris_data2
import numpy as np
from scipy import stats

if __name__ == '__main__':
  (train_x, train_y), (test_x, test_y) = iris_data2.load_data()

  xx = iris_data2.train_input_fn(train_x, train_y)
  
  with tf.Session() as sess:
    print(stats.describe(train_x["PetalWidth"]))
    print('----')

    for ii in np.arange(20):
      vv = sess.run(xx)
      print(stats.describe(vv[0]['PetalWidth']))

