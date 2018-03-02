"""Iteration unus: a first attempt at a CNN, that uses lower-level API than
iteration-duo.

In the repo solely for journaling purposes.

Does not work due to dimension-related errors that are too stubborn to fix
elegantly, but has been used as the basis for the architecture of
iteration-duo. Has also been used to get an understanding of tf, and the bugs
that exist in various versions of tf running on MacOSX.

Very light adaptation of:
    http://adventuresinmachinelearning.com/python-tensorflow-tutorial/

Intended working directory: "."
"""

import pickle

import tensorflow as tf
from sklearn.model_selection import train_test_split


class MacOSFile():

    """Workaround for issue 24658 (https://bugs.python.org/issue24658).

    Source: https://stackoverflow.com/questions/31468117/python-3-can-pickle- \
    handle-byte-objects-larger-than-4gb.
    """

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size),
                # \ end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size),
                  end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size


def pickle_dump(obj, file_path):
    """Wrapper of pickle.dump"""
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_path):
    """Wrapper of pickle.load"""
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))


def run_cnn(lot_name, subset_size=-1, learning_rate=0.0001, num_epochs=10,
            batch_size=50, testing_pct=0.25):
    base_path = "data/pickles/" + lot_name
    features = pickle_load(base_path + "_X.npy")
    labels = pickle_load(base_path + "_y.npy")

    # Loading the total number of examples we want;
    # total = training + testing + validation
    if subset_size == -1:
        subset_size = features.shape[0]
    train_X, test_X, train_y, test_y = train_test_split(features[:subset_size],
                                                        labels[:subset_size],
                                                        shuffle=True,
                                                        test_size=testing_pct)

    # Placeholders for the batch inputs.
    for_trg_X = tf.placeholder(tf.float32, [batch_size, 32, 32, 3])
    for_trg_y = tf.placeholder(tf.float32, [batch_size, 1])
    for_tst_X = tf.placeholder(tf.float32, [test_X.shape[0], 32, 32, 3])
    for_tst_y = tf.placeholder(tf.float32, [test_y.shape[0], 1])

    # First convolutional layer:
    # Input is three deep, output is 32 deep.
    conv_1 = create_new_conv_layer(for_trg_X, 3, 32, [5, 5], [2, 2],
                                   name="conv_1")

    # Second convolutional layer:
    # Input is 32 deep, output is 64 deep.
    conv_2 = create_new_conv_layer(conv_1, 32, 64, [5, 5], [2, 2],
                                   name="conv_2")

    # Flattening output of conv_2 for input into the fully-connected layer.
    fcl_input = tf.reshape(conv_2, tf.convert_to_tensor([-1, 8 * 8 * 64],
                                                        dtype=tf.int32))

    # Initializing random N(0, 0.009) weights for the fcl.
    # The fcl has a 1000 neurons.
    weights_1 = tf.Variable(tf.truncated_normal([8 * 8 * 64, 1000],
                                                stddev=0.03),
                            name="weights_1")
    # Initializing random N(0, 0.001) biases for the fcl.
    biases_1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01),
                           name="biases_1")
    # We pass the input volume through the fcl model and then apply an
    # activation function to it.
    dense_1 = tf.nn.relu(
            tf.matmul(fcl_input, weights_1)
            + biases_1)
    # We pass the results/"scores" to a softmax function.
    weights_2 = tf.Variable(tf.truncated_normal([1000, 1], stddev=0.03),
                            name="weights_2")
    biases_2 = tf.Variable(tf.truncated_normal([1], stddev=0.01),
                           name="biases_2")
    dense_2 = tf.matmul(dense_1, weights_2) + biases_2
    y_pred = tf.nn.softmax(dense_2)

    # We now calculate a measure of the prediction error.
    cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=dense_2,
                                                    labels=for_trg_y)
            )
    # This measure is to be used to create an optimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
            cross_entropy)

    # The operation for measuring current accuracy.
    correct_prediction = tf.equal(tf.argmax(for_trg_y, 1),
                                  tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # setup the initialization operator.
    init_op = tf.global_variables_initializer()

    # The below is a recording variable that stores the current accuracy.
    tf.summary.scalar("accuracy", accuracy)
    # We store the summary in the given folder.
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("ml_summaries/")

    with tf.Session() as sess:
        sess.run(init_op)
        num_batches = int(train_X.shape[0] / batch_size)
        for epoch in range(num_epochs):
            avg_cost = 0
            for i in range(num_batches):
                curr_start = i * batch_size
                curr_end = (i + 1) * batch_size
                curr_X, curr_y = get_next_batch(train_X, train_y,
                                                curr_start, curr_end)
                _, c = sess.run([optimizer, cross_entropy],
                                feed_dict={for_trg_X: curr_X,
                                           for_trg_y: curr_y})
                avg_cost += c / num_batches
            test_acc = sess.run(accuracy,
                                feed_dict={for_tst_X: test_X,
                                           for_tst_y: test_y})
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost),
                  " test accuracy: {:.3f}".format(test_acc))
            summary = sess.run(merged,
                               feed_dict={for_tst_X: test_X,
                                          for_tst_y: test_y})
            writer.add_summary(summary, epoch)

        print("\nTraining complete.")
        writer.add_graph(sess.graph)
        print(sess.run(accuracy, feed_dict={for_tst_X: test_X,
                       for_tst_y: test_y}))


def create_new_conv_layer(input_data, num_input_channels, num_filters,
                          filter_shape, pool_shape, name):
    # Set up the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                       num_filters]

    # InitialiZe weights and bias for the filter.
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                          name=name+"_W")
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+"_b")

    # Setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding="SAME")
    out_layer += bias
    out_layer = tf.nn.relu(out_layer)

    # Now we perform max pooling.
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides,
                               padding='SAME')

    return out_layer


def get_next_batch(train_X, train_y, curr_start, curr_end):
    return train_X[curr_start:curr_end], train_y[curr_start:curr_end]


if __name__ == "__main__":
    run_cnn("primis", 3000)
