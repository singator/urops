# We start at the beginning. Let us read in the data from pX and pY npy files.
# Next, we split into 80/10/10 training/validation/test sets.

import tensorflow as tf
import numpy as np
import pandas as pd
import pickle, os, time
from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit

global dropout_rate, learning_rate

def cnn_model_fn(features, labels, mode):
    """cnn_model_fn creates an Estimator from the inputs.

    Arguments:
        features: a tensor of dimensions [batch_size, 32*32, 3].
        labels: a tensor of dimensions [batch_size] representing the
        corresponding occupancy status for each example spot.
        mode:
            TRAIN: training mode.
            EVAL: evaluation mode.
            PREDICT: inference mode.
    Returns:
        An Estimator in the image of TensorFlow's Estimator API, which contains
        the predictions, loss, and a training operation.
    """

    # First convolutional+pooling layer.
    conv1 = tf.layers.conv2d(
        inputs=features["x"],
        filters=32,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=2,
            strides=2)

    # Second sconvolutional+pooling layer.
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=3,  # Works better than 5.
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=[2, 2],
            strides=2)

    # Dense layer -- dropout enabled if mode == "TRAIN".
    pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])
    dense = tf.layers.dense(
            inputs=pool2_flat,
            units=1024,
            activation=tf.nn.relu)
    dropout = tf.layers.dropout(
            inputs=dense,
            rate=dropout_rate,
            training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1, name='test'),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate loss (for both TRAIN and EVAL modes).
    loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels,
            logits=logits)

    # Logging Hooks
    tensors_to_log = {'loss': loss, 'step': tf.train.get_global_step()}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, 
       every_n_iter=10)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate)
        train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op, 
                training_hooks=[logging_hook])

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                    labels=labels,
                    predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops)

if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.INFO)
    dropout_rate=0.25
    learning_rate = 0.0001

    # Read in the pX and pY data.
    # In total there are 105843 examples.
    fpathX = '../data/pX.npy'
    fpathY = '../data/pY.npy'

    with open(fpathX, "rb") as f:
        pX = pickle.load(f)
        pX = pX.astype(np.float16)
    with open(fpathY, "rb") as f:
        pY = pickle.load(f)
        pY = pY.astype(np.int32)

    # Split the data into training, validation and test sets. For now, we use #
    # percentage 80/10/10.
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2228)
    for tr_id,te_id in split.split(pX, pY):
      train_x = pX[tr_id]
      train_y = pY[tr_id]

      test_x =  pX[te_id]
      test_y = pY[te_id]

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=2229)
    for tr_id,te_id in split.split(test_x, test_y):
      val_x = test_x[tr_id]
      val_y = test_y[tr_id]

      test_x = test_x[te_id]
      test_y = test_y[te_id]

    # train_x = pX[0:100]
    # train_y = pY[0:100]

    # reset graph first
    tf.reset_default_graph()

    # Construct estimator first
    now = datetime.now()
    logdir = "tf_logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"

    classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,
            model_dir=logdir)

    # Set up logging for predictions.
    # Log the values in the "Softmax" tensor with label "probabilities".
    # tensors_to_log = {"probabilities": "softmax_tensor", 'classes': 'test'}
    # tensors_to_log = {'classes': 'test'}
    # logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=3)

    # Train the model.
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_x}, y=train_y, batch_size=10,
            num_epochs=None, shuffle=False)

    # Train 20 steps
    #classifier.train(input_fn=train_input_fn, steps=4, hooks=[logging_hook])
    classifier.train(input_fn=train_input_fn, steps=100)
