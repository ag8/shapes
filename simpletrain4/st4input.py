from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from random import randint

import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


# The image size to crop the image to.
# Keep at 100 to avoid any cropping;
# do not set this value below 80
IMAGE_SIZE = 100

# Global constants describing the MSHAPES data set.
NUM_CLASSES = 2  # Binary classification
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000



def read_mshapes(filename_queue):
    """
    Reads a pair of MSHAPE records from the filename queue.

    :param filename_queue: the filename queue of lock/key files.
    :return: a duple containing a correct example and an incorrect example
    """
    file_pair1 = filename_queue.dequeue()
    file_pair2 = filename_queue.dequeue()

    _, key_image = decode_mshapes(file_pair1[1])
    _, lock_image = decode_mshapes(file_pair1[0])
    _, wrong_key_image = decode_mshapes(file_pair2[1])  # The key from the next pair in the queue
    # will not match the current lock

    # Combine images to make a correct example and an incorrect example
    correct_example = tf.concat([lock_image, key_image], axis=0)
    wrong_example = tf.concat([lock_image, wrong_key_image], axis=0)

    # Return the examples
    return correct_example, wrong_example



def decode_mshapes(file_path):
    """
    Decodes an MSHAPE record.

    :param file_path: The filepath of the png
    :return: A duple containing 0 and the decoded image tensor
    """

    # read the whole file
    serialized_record = tf.read_file(file_path)

    # decode everything into int32
    image = tf.image.decode_png(serialized_record)

    return 0, image



def inputs(eval_data, data_dir, batch_size):
    """
    Constructs the input for MSHAPES.

    :param eval_data: boolean, indicating if we should use the training or the evaluation data set
    :param data_dir: Path to the MSHAPES data directory
    :param batch_size: Number of images per batch

    :return:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 6] size
        labels: Labels. 1D tensor of [batch_size] size.
    """

    filequeue = tf.FIFOQueue(capacity=100000, dtypes=[tf.string, tf.string])  # FIXME: Use RandomShuffleQueue instead!!
    enqueues = []

    if not eval_data:
        print("Not eval data")
        for i in xrange(1, 5000):  # TODO: First of all, this should go to (at least) 30000.
                                # The reason it's at 5000 is that currently, we're
                                # individually enqueueing images. Instead, we should
                                # use enqueue_many with an inline for loop, which
                                # should building up the queue much faster.
            # print(i)
            if (i % 1000 == 1):
                print(i)
            lock = os.path.join(data_dir, 'images/%d_L.png' % i)
            key = os.path.join(data_dir, 'images/%d_K.png' % i)

            a = filequeue.enqueue([lock, key])
            enqueues.append(a)

        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
        print("Ok")
    else:
        for i in xrange(30001, 49500):
            lock = os.path.join(data_dir, 'images/%d_L.png' % i)
            key = os.path.join(data_dir, 'images/%d_K.png' % i)

            a = filequeue.enqueue([lock, key])
            enqueues.append(a)

        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

    print("Finished enqueueing")

    # Get the correct and incorrect examples from files in the filename queue.
    c, w = read_mshapes(filequeue)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return enqueues, _generate_image_and_label_batch(c, 1,  # FIXME: Also add incorrect examples to the batch
                                                     min_queue_examples, batch_size,
                                                     shuffle=False)



def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.
    Args:
      image: 3-D Tensor of [height, width, 3] of type.float32.
      label: 1-D Tensor of type.int32
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
      batch_size: Number of images per batch.
      shuffle: boolean indicating whether to use a shuffling queue.
    Returns:
      images: Images. 4D tensor of [batch_size, height, width, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.summary.image('images', images)

    print("Images dimensions: ", images.get_shape())

    return images, tf.reshape(label_batch, [batch_size])
