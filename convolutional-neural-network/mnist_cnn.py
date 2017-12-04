import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 128
test_size = 256
img_size = 28
num_classes = 10

# randonmized the initial weight for the layer
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


# build the training model, using 4 layers convolution network

def model(X, W1, W2, W3, W4, W_O, keep_prob_conv, keep_prob_hidden):
    conv1 = tf.nn.conv2d(X, W1,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv1_a = tf.nn.relu(conv1)  # activation function

    conv1 = tf.nn.max_pool(conv1_a, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob=keep_prob_conv)

    conv2 = tf.nn.conv2d(conv1, W2,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv2_a = tf.nn.relu(conv2)

    conv2 = tf.nn.max_pool(conv2_a, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob=keep_prob_conv)

    conv3 = tf.nn.conv2d(conv2, W3,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv3_a = tf.nn.relu(conv3)

    fc_layer = tf.nn.max_pool(conv3_a, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

    fc_layer = tf.reshape(fc_layer, [-1, W4.get_shape().as_list()[0]])

    fc_layer = tf.nn.dropout(fc_layer, keep_prob=keep_prob_conv)

    output_layer = tf.nn.relu(tf.matmul(fc_layer, W4))

    output_layer = tf.nn.dropout(output_layer, keep_prob=keep_prob_hidden)

    result = tf.matmul(output_layer, W_O)

    return result


mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

trX = trX.reshape(-1, img_size, img_size, 1)

teX = teX.reshape(-1, img_size, img_size, 1)

X = tf.placeholder("float", [None, img_size, img_size, 1])
Y = tf.placeholder("float", [None, num_classes])

W1 = init_weights([3,3,1,32])
W2 = init_weights([3,3,32, 64])
W3 = init_weights([3,3,64,128])
W4 = init_weights([128 * 4 * 4, 625])
W_O = init_weights([625, num_classes])

keep_prob_conv = tf.placeholder("float", name='prob_conv')
keep_prob_hidden = tf.placeholder("float", name='prob_hidden')

PY_X = model(X, W1, W2, W3, W4, W_O, keep_prob_conv, keep_prob_hidden)

Y_ = tf.nn.softmax_cross_entropy_with_logits(logits=PY_X, labels=Y)

cost = tf.reduce_mean(Y_)

optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

predict_op = tf.argmax(PY_X, 1)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(1000):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX) + 1, batch_size))

        for start, end in training_batch:
            sess.run(optimizer, feed_dict={X: trX[start:end],
                                           Y: trY[start:end],
                                           keep_prob_conv: 0.8,
                                           keep_prob_hidden: 0.5})

        test_indices = np.arange(len(teX))  # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op,
                                  feed_dict={
                                      X: teX[test_indices],
                                      Y: teY[test_indices],
                                      keep_prob_conv: 1.0,
                                      keep_prob_hidden: 1.0}
                                  )))


