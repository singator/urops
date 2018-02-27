"""iteration-duo: a second, working attempt at CNN.

This docstring and the ones to be below it will be updated after our subsequent
meeting.

Built upon
    https://github.com/aymericdamien/TensorFlow-Examples/
after consideration of the architecture of iteration-unus.

Intended working directory: "."
"""

from __future__ import division, print_function, absolute_import

import os

os.chdir("/Users/nurmister/Documents/academic/urops/")

import tensorflow as tf

from retrieval import data_manager

# Data Parameters
lot_name = "primis"
subset_size = -1
testing_pct = 0.25
# TODO: consider implications of randomization of test and train, each time.
# i.e. consider if it would be better to pickle one configuration?
train_X, test_X, train_y, test_y = data_manager.create_data(lot_name,
                                                            subset_size,
                                                            testing_pct)

# Training parameters
learning_rate = 0.001
num_steps = 2000
batch_size = 128

# Network parameters
num_input = 1024
num_classes = 2
dropout = 0.25


# Create the neural network
def conv_net(x_dict, num_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        x = tf.cast(x_dict['images'], tf.float32)

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, num_classes)

    return out


# Define the model function (following the TF Estimator Template)
def model_fn(features, labels, mode):
    # Because Dropout has different behavior at training and prediction time,
    # we need to create 2 distinct computation graphs that still share the
    # same weights.
    logits_train = conv_net(features, num_classes, dropout, reuse=False,
                            is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True,
                           is_training=False)

    # Predictions.
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probs = tf.nn.softmax(logits_test)

    # If prediction mode, early return.
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # Define loss and optimizer.
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model.
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, &c.
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs


# Build the estimator.
model = tf.estimator.Estimator(model_fn, model_dir="ml_algorithms/logs/1")

# Define the input function for training.
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': train_X}, y=train_y,
    batch_size=batch_size, num_epochs=None, shuffle=True)
# Train the model.
model.train(input_fn, steps=num_steps)

# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': test_X}, y=test_y,
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])
