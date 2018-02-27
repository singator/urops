"""iteration-tribus: a third attempt at CNN, built on the layers API.

Inspired by (only inspired -- the tutorial, directly applied to our problem, 
produced horrible results):
    https://www.tensorflow.org/tutorials/layers
Has the same architecture as iteration-duo.

Current best evaluation accuracies (4 s.f.):
    primis: 99.78%
    secondus: 99.52%
    solum: 99.87%
All of the above are better than iteration-duo.
The splits that resulted in the above are on Floydhub and Google Drive.

Intended working directory: "./ml_algorithms"
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import os
import time
import argparse

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# The following hyperparameters are global variables because `cnn_model_fn()`
# cannot accept any arguments except those currently listed, in order to
# conform with the `Estimator` API.
# Both of the below are set when the function is called from the command line--
# i.e. they are flags.
global dropout_rate, learning_rate


def find_num_steps(name, batch_size, testing_pct):
    """Depending on the batch_size, returns the minimum number of steps
    required to train the model on each training example once."""
    arr = pickle_load("data/pickles_2/" + name + "_y.npy")
    num_ex = (1 - testing_pct) * arr.shape[0]
    return int(num_ex / batch_size) + 1


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

    global dropout_rate, learning_rate

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
        "classes": tf.argmax(input=logits, axis=1),
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

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate)
        train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                    labels=labels,
                    predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops)


def main(lot_name, testing_pct, batch_size, num_steps):
    """Load train and test sets, create Estimator, set up logging hooks, and
    train and evaluate the Estimator.

    Arguments:
        lot_name: {"primis", "secondus", "solum"}.
        testing_pct: (0, 1).
        batch_size: Preferably powers of 2.
        num_steps: Depending on the batch_size, is set to the minimum number of
        steps required to train the model on each training example once.
    Returns:
        Prints and saves logs, and saves model checkpoints.
    """

    tmp = pickle.load(tmp, "../data/split_" + lot_name + ".npy")
    train_X, test_X, train_y, test_y = tmp[0], tmp[1], tmp[2], tmp[3]

    # Create the Estimator
    classifier = tf.estimator.Estimator(
            model_fn=cnn_model_fn,
            model_dir="logs/" + lot_name + "_" +
            time.strftime("%d-%m") + "_" + time.strftime("%H-%M"))

    # Set up logging for predictions.
    # Log the values in the "Softmax" tensor with label "probabilities".
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)

    # Train the model.
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_X},
            y=train_y,
            batch_size=batch_size,
            num_epochs=None,  # Only one epoch is executed.
            shuffle=False)
    classifier.train(
            input_fn=train_input_fn,
            steps=num_steps,
            hooks=[logging_hook])

    # Evaluate the model and print results.
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": test_X},
            y=test_y,
            num_epochs=1,
            shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    global dropout_rate, learning_rate

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--batchsize", type=int)
    parser.add_argument("--testingpct", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--learning", type=float)

    args = parser.parse_args()
    dropout_rate = args.dropout
    learning_rate = args.learning

    main(
            args.name,
            args.testingpct,
            args.batchsize,
            find_num_steps(
                    args.name,
                    args.batchsize,
                    args.testingpct))
