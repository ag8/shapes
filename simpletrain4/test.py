#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

from six.moves import xrange

from simpletrain4 import st4
from utils import *

if get_hostname() == "lebennin-vm":
    print("Привет!")

# Download and extract the dataset if it's missing
print("Setting up dataset...")
# maybe_download_and_extract()
print("Done.")

# Run some checks on the dataset to make sure it's correct
print("Running tests...")
verify_dataset()
print("All tests passed.")

# Clean up directories
print("Cleaning up directories...")
if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
tf.gfile.MakeDirs(FLAGS.train_dir)
print("Done.")



# Get images and labels for MSHAPES
print("Setting up getting batches and labels")
images_batch, labels_batch = st4.inputs(eval_data=False)
print("Got two batches")

print("Image batch shape: ")
print(images_batch.get_shape())
print("Labels batch shape:")
print(labels_batch.get_shape())


with tf.Session(config=tf.ConfigProto(log_device_placement=False, operation_timeout_in_ms=60000)) as sess:
    print("Actually running now")

    print("Initializing global variables")
    tf.set_random_seed(42)
    tf.global_variables_initializer().run()
    print("Finished")

    print("Starting coordinator and queue runners")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    print("Ok")

    for i in xrange(0, 5):
        print("blah")

        # Get an image tensor and print its value.
        print("Getting image tensor")
        image_tensor = sess.run([images_batch])
        print("Got image tensor")
        print(image_tensor[0][0, 0, 0])

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)
