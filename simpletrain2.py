from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

import simpletrain2input as st2i
from shape_generation import Flags



def main(_):
    # Make logging very verbose
    tf.logging.set_verbosity(tf.logging.DEBUG)


    sess = tf.Session()

    # Get a filename queue
    filequeue = st2i.database_to_filename_queue(Flags.image_path, shuffle=True)

    print("File queue: ", filequeue)

    # Now, from the filename queue, get queues of combined shape images and labels
    image_batch, label_batch = st2i.filename_queue_to_image_and_label_queue(filequeue)



    # Generate the data placeholders

    x = tf.placeholder(tf.float32, [None, 60000])
    W = tf.Variable(tf.zeros([60000, 2]))
    b = tf.Variable(tf.zeros([2]))
    y = tf.matmul(x, W) + b


    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 2])

    cross_entropy = tf.reduce_mean(
                            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.AdamOptimizer(0.05).minimize(cross_entropy)



    # Initialize all the variables/queues

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)


    # Train!
    for _ in range(1000):
        print("Image batch shape:", image_batch.get_shape())

        batch_xs, batch_ys = sess.run([image_batch, label_batch])
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        # Test trained model
        # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # print(sess.run(accuracy, feed_dict={x: mnist.test.images,
        #                                     y_: mnist.test.labels}))



    coord.request_stop()
    coord.join(threads)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()

tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)