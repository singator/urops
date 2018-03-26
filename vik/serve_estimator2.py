
import tensorflow as tf
import numpy as np

if __name__ == '__main__':

  eg = []
  example = tf.train.Example( 
    features=tf.train.Features( 
      feature={
        "SepalLength":
tf.train.Feature(float_list=tf.train.FloatList(value=[4.7])),
        "PetalLength":
tf.train.Feature(float_list=tf.train.FloatList(value=[1.6])),
        "SepalWidth":
tf.train.Feature(float_list=tf.train.FloatList(value=[3.2])),
        "PetalWidth":
tf.train.Feature(float_list=tf.train.FloatList(value=[0.2]))
      }
    )
  )
  eg.append(example.SerializeToString())
  eg.append(example.SerializeToString())

  serialised_eg = example.SerializeToString()
  fmd = 'tf_logs/test/1522080061'
  
  with tf.Session() as sess:
    # tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], fmd)
    predictor=tf.contrib.predictor.from_saved_model(fmd)
    # out = predictor({"inputs": [serialised_eg]})
    out = predictor({"inputs": eg})
    print(np.round(out['scores'], 3))
