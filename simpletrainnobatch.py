from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
from glob import glob
from random import randint
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

import Flags


FLAGS = None



def index_the_database_into_queue(image_path, shuffle):
    """Indexes av4 database and returns two lists of filesystem path: ligand files, and protein files.
        Ligands are assumed to end with _ligand.av4, proteins should be in the same folders with ligands.
        Each protein should have its own folder named similarly to the protein name (in the PDB)."""


    # STEP 1: GET LIST OF KEY AND LOCK FILES


    key_file_list = []
    lock_file_list = []

    print("Number of keys:", len(glob(os.path.join(image_path + '', "*[_]*L.png"))))

    for key_image in glob(os.path.join(image_path + '', "*[_]*L.png")):
        # print(key_image)

        lock_image = key_image.replace('L', 'K')

        if os.path.exists(lock_image):
            key_file_list.append(key_image)
            lock_file_list.append(lock_image)
        else:
            print("Could not find lock for key ", key_image)

    index_list = range(len(key_file_list))
    examples_in_database = len(index_list)

    if examples_in_database == 0:
        raise Exception('No files found in the image path:', image_path)

    print("Number of indexed key-lock pairs in the database:", examples_in_database)

    # create a filename queue (tensor) with the names of the keys and locks
    index_tensor = tf.convert_to_tensor(index_list, dtype=tf.int32)
    key_files = tf.convert_to_tensor(key_file_list, dtype=tf.string)
    lock_files = tf.convert_to_tensor(lock_file_list, dtype=tf.string)

    print("Lengths:", index_tensor.get_shape(), ", ", key_files.get_shape(), ", ", lock_files.get_shape())



    # STEP 2: MAKE LIST OF MATCHING/NOT MATCHING ISNTRUCTIONS

    probability_of_match = 0.5
    match_or_not_list = np.random.choice([0, 1], size=(50000,), p=[1 - probability_of_match, probability_of_match])

    match_or_not = tf.convert_to_tensor(match_or_not_list, dtype=tf.bool)





    # filename_queue = tf.train.slice_input_producer([index_tensor, key_files, lock_files], num_epochs=None,
    #                                                shuffle=shuffle)
    # filename_queue = [index_list, key_file_list, lock_file_list]

    rsq = tf.RandomShuffleQueue(50000, 0, [tf.int32, tf.string, tf.string, tf.bool], shapes=[[], [], [], []])
    do_enqueues = rsq.enqueue_many([index_tensor, key_files, lock_files, match_or_not])

    index, key_file, lock_file, match = rsq.dequeue()

    return do_enqueues, index, key_file, lock_file, match, examples_in_database



def random_but_not((min, max), avoid):
    """Chooses a random number in a range from min to max (inclusive), but avoiding the number avoid."""
    chosen = avoid

    while chosen == avoid:
        chosen = randint(min, max)

    return chosen



def image_and_label_queue(batch_size, num_threads, index, key_file, lock_file, match, train=True):
    """Creates shuffle queue for training the network"""

    print("Getting index")

    lock_image_num = index

    print("Got index")

    print("Getting lock image")
    lock_image = tf.image.decode_png(lock_file)
    key_image = None
    print("Got lock image")

    match_or_not = match  # Determine whether the key should match the lock or not
    print("Got matchornot")

    if match_or_not == 1:
        print("Setting up true matchornot")
        key_image = tf.image.decode_png(key_file)
        print("Got matching key image")
    else:
        print("Setting up false matchornot")
        new_key_file = key_file + "abc"
        key_image = tf.image.decode_png(new_key_file)
        print("Got non-matching key image")

    print("Combining images")
    combined_image = tf.stack([lock_image, key_image], axis=1)
    print("Combined images")

    print("Getting label")
    label = match_or_not  # The label says whether the lock and the key actually match
    print("Got label")

    # print("Batching")
    # create a batch of proteins and ligands to read them together
    # num, label_batch, image_batch = tf.train.batch([lock_image_num, label, combined_image], batch_size, num_threads=num_threads,
                                       # capacity=batch_size * 3, dynamic_pad=True)

    # print("Batch shape: ", str(tf.shape(multithread_batch)))
    # print("Finished batching")

    return (lock_image_num, label, combined_image)



def main(_):
    sess = tf.Session()

    # Get a filename queue
    do_enqueues, index_queue, key_file_queue, lock_file_queue, match, examples_in_database = index_the_database_into_queue(Flags.image_path, shuffle=True)

    print("Gotten filename queue.")

    # fq = sess.run(filename_queue)
    # print("FQ: ", fq)

    # create a custom shuffle queue
    num, image_batch, label_batch = image_and_label_queue(batch_size=Flags.batch_size,
                                                          num_threads=Flags.num_threads,
                                                          index=index_queue,
                                                          key_file=key_file_queue,
                                                          lock_file=lock_file_queue,
                                                          match=match)

    print("Obtained batch")

    # Import data

    x = tf.placeholder(tf.float32, [None, None])
    W = tf.Variable(tf.zeros([100, 2]))
    b = tf.Variable(tf.zeros([2]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 2])

    cross_entropy = tf.reduce_mean(
                            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.AdamOptimizer(0.05).minimize(cross_entropy)


    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)


    print("Starting training!")

    # Train
    for _ in range(1000):
        print("In training loop.")

        print("Running do_enqueues")
        sess.run(do_enqueues)
        print("Finished running do_enqueues")
        print("Evaluating image batches")
        batch_xs, batch_ys = sess.run([image_batch, label_batch])
        print("image: ", batch_xs)
        print("Finished evaluating image batches")

        print("Gettting shape of x batch")
        shape = tf.shape(batch_xs)
        print("Got shape of x batch")
        print("Evaluating shape of x batch")
        k = sess.run(shape)
        print("Evaluated shape of x batch")
        print("Shape: ", k)

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
