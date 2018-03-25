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
import pandas as pd
from scipy import stats

if __name__ == '__main__':

  x = np.zeros(100, np.uint8)
  y = np.ones(100, np.uint8)
  x[9] = 1
  xd = pd.DataFrame([x1 for x1 in x],
                    columns=['x'])

  xx  = iris_data2.train_input_fn(xd, y)
  
  with tf.Session() as sess:
    for ii in np.arange(100):
      vv = sess.run(xx)[0]['x']
      vvm = np.max(vv)
      if(vvm != 1):
        print('1 missing')
