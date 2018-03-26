from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import iris_data2
import pandas as pd

from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')
parser.add_argument('--n_epochs', default=5, type=int, help='number of epochs')

def train_input_fn(features, labels):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    """
    The following combination returns a tuple of length 120, with a dict for 
    trainX and array for trainY:
    dataset = dataset.shuffle(1000).repeat().batch(120)
    return dataset.make_one_shot_iterator().get_next()
  
    The following combination returns an error that I cannot solve:
    dataset = dataset.shuffle(1000).repeat().batch(120)
    return dataset
    """
    # Shuffle, repeat, and batch the examples.
    # dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    # dataset = dataset.shuffle(120).repeat()
    #dataset = dataset.shuffle(1000).repeat().batch(100)

    dataset = dataset.repeat().batch(105)
    return dataset.make_one_shot_iterator().get_next()

def eval_input_fn(features, labels):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    # dataset = dataset.repeat().batch(22)

    # Return the read end of the pipeline.
    return dataset.batch(len(features))

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)

  args = parser.parse_args()

  tf.reset_default_graph()

  # Fetch the data
  (train_x, train_y), (test_x, test_y) = iris_data2.load_data()
  all_y = pd.concat([test_y, train_y], ignore_index=True)
  all_x = pd.concat([test_x, train_x], ignore_index=True)

  # Split the data
  split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=203)
  for tr_id,te_id in split.split(all_x, all_y):
    train_x = all_x.iloc[tr_id]
    test_x = all_x.iloc[te_id]

    train_y = all_y.iloc[tr_id]
    test_y = all_y.iloc[te_id]

  split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=204)
  for tr_id,te_id in split.split(test_x, test_y):
    val_x = test_x.iloc[tr_id]
    test_x = test_x.iloc[te_id]

    val_y = test_y.iloc[tr_id]
    test_y = test_y.iloc[te_id]

  # Create estimator
  # Feature columns describe how to use the input.
  my_feature_columns = []
  for key in train_x.keys():
      my_feature_columns.append(tf.feature_column.numeric_column(key=key))

  now = datetime.now()
  logdir = "tf_logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"

  # Build 2 hidden layer DNN with 10, 10 units respectively.
  classifier = tf.estimator.DNNClassifier(
      feature_columns=my_feature_columns,
      # Two hidden layers of 10 nodes each.
      hidden_units=[10, 10],
      # The model must choose between 3 classes.
      n_classes=3, 
      model_dir=logdir)

  # Train the Model.
  for ii in range(args.n_epochs):

    classifier.train(
        input_fn=lambda:train_input_fn(train_x, train_y), steps=1)

    if((ii > 1) & (ii % 10 == 0)):
      eval_result = classifier.evaluate(input_fn=lambda:eval_input_fn(val_x,
        val_y), steps=1) 

 # Check test set accuracy:
  eval_result2 = classifier.evaluate(input_fn=lambda:eval_input_fn(test_x,
    test_y), steps=1) 
  print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result2))

  f_spec = tf.feature_column.make_parse_example_spec(my_feature_columns)
  input_rec_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(f_spec)

 # Export the model for serving. See serve_estimator2.py
  export_dir = classifier.export_savedmodel(export_dir_base="tf_logs/test",
      serving_input_receiver_fn=input_rec_fn)
  print(export_dir)
