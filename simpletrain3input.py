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



def read_mshapes(lock_queue, key_queue, match_queue):
    """
    This function returns a combined lock/key image and a label
    given a lock queue, a key queue, and a queue that stores
    whether the lock and the key match.

    :param lock_queue: a filequeue of the lock image files
    :param key_queue: a filequeue of the key image files
    :param match_queue: a queue that states whether the lock and the key match

    :return: an MShapeRecord containing the combined image and the label
    """


    class MShapeRecord(object):
        pass


    # Create a WholeFileReader to read in the images
    image_reader = tf.WholeFileReader()

    # Read a whole file from the queue.
    # The first returned value is the filename,
    # which we can safely ignore
    _, lock_image_file = image_reader.read(lock_queue)
    _, key_image_file = image_reader.read(key_queue)

    # Decode the image as a PNG.
    # This will turn the image into a tensor,
    # which we can then use in training.
    # TODO: Read in as grayscale (so as not to learn color matching)
    lock_image = tf.image.decode_png(lock_image_file)
    key_image = tf.image.decode_png(key_image_file)

    # Concatenate the lock tensor and the key tensor into one tensor
    image = tf.concat([lock_image, key_image], axis=1)

    # Write the image/label to an MShapeRecord-type object
    read_input = MShapeRecord()

    read_input.uint8image = image
    read_input.label = match_queue.dequeue()
    # read_input.label = tf.convert_to_tensor([1])
    print("Label in read function:", read_input.label)
    # read_input.label.set_shape([1])
    # print("Label in read function:", read_input.label)

    return read_input


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

    match_or_not = np.random.choice(2, 100000)

    if not eval_data:
        locks = []
        keys = []

        for i in xrange(1, 30000):
            match = match_or_not[i]

            locks.append(os.path.join(data_dir, 'images/%d_L.png' % i))

            if match == 0:
                j = randint(1, 29999)  # TODO: Make sure i is not randomly picked by accident
            else:
                j = i + 0

            keys.append(os.path.join(data_dir, 'images/%d_K.png' % j))

        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        locks = []
        keys = []

        for i in xrange(30001, 49500):
            match = match_or_not[i]

            locks.append(os.path.join(data_dir, 'images/%d_L.png' % i))

            if match == 0:
                j = randint(30001, 49500)  # TODO: Make sure i is not randomly picked by accident
            else:
                j = i + 0

            keys.append(os.path.join(data_dir, 'images/%d_K.png' % j))

        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

    for f in locks:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    for f in keys:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    lock_queue = tf.train.string_input_producer(locks)
    key_queue = tf.train.string_input_producer(keys)
    match_queue = tf.FIFOQueue(capacity=10000, dtypes=tf.uint8)
    match_queue.enqueue_many(match_or_not)

    # Read examples from files in the filename queue.
    read_input = read_mshapes(lock_queue, key_queue, match_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    print("Acquired read_input.")

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           height, width)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(resized_image)

    print("Got the image and labels; now, setting the shape of tensors.")

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    print("Finished setting shapes of tensors.")

    print("Label:", read_input.label)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
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
