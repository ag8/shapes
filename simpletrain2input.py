import os
from glob import glob

import tensorflow as tf

import numpy as np

import Flags



def database_to_filename_queue(images_path, shuffle=True):
    """Creates a RandomShuffleQueue of filenames based on the directory for images.

        Args:
          images_path: The directory containing all
                    of the images to be placed into
                    the filename queue.

          shuffle: (Optional.) Whether the order of
                    files should be shuffled.
                    Defaults to True (shuffle)

    """

    # First, we get a list of lock files and key files.

    key_file_list = []
    lock_file_list = []

    # All key image paths are in the form {images_path}/[n]_K.png
    # where [n] is some integer representing the ID of the image.
    key_image_path = os.path.join(images_path + '', "*[_]*K.png")

    print("Number of keys:", len(glob(key_image_path)))  # TODO: Implement a better logging system

    # Cycle through all files in the key_image_path form
    for key_image in glob(key_image_path):

        # Get the corresponding lock image for the key image.
        # Lock images are in the form {images_path}/[n]_L.png;
        # that is, the only difference from the key image path
        # is the K->L substitution.
        lock_image = key_image.replace('K', 'L')

        # Append the lock and key files to their respective lists (if the files exist)
        if os.path.exists(lock_image):
            key_file_list.append(key_image)
            lock_file_list.append(lock_image)
        else:
            print("Could not find lock for key ", key_image)
            print("(Skipping both)")

    # index_list is just a list of integers from 0 to the number of images minus one
    index_list = range(len(key_file_list))

    examples_in_database = len(index_list)

    # If there are no images found, say so
    if examples_in_database == 0:
        raise Exception('No files found in the image path:', images_path)

    print("Number of indexed key-lock pairs in the database:", examples_in_database)

    # Now, we create a list of booleans, which will tell us whether
    # we should match the lock to its key or not. This is used in
    # later methods, which actually read and combine the images.
    probability_of_match = 0.5  # The probability that we will match the key to its lock correctly (for training)
    match_or_not_list = np.random.choice([0, 1],
                                         size=(len(key_file_list),),
                                         p=[1 - probability_of_match, probability_of_match])

    # Now, create tensors from the index, key file, lock file, and matching lists
    index_tensor = tf.convert_to_tensor(index_list, dtype=tf.int32)
    key_files = tf.convert_to_tensor(key_file_list, dtype=tf.string)
    lock_files = tf.convert_to_tensor(lock_file_list, dtype=tf.string)
    match_or_not = tf.convert_to_tensor(match_or_not_list, dtype=tf.bool)

    # Now, we generate a queue based on the four tensors we made from the lists above
    filename_queue = tf.train.slice_input_producer([index_tensor, key_files, lock_files, match_or_not], num_epochs=None,
                                                   shuffle=shuffle)

    return filename_queue



def filename_queue_to_image_and_label_queue(filename_queue, epoch_counter, batch_size=Flags.batch_size,
                                            num_threads=Flags.num_threads):
    """Creates a queue of images and labels based on the filename/matching queue

        Args:
          filename_queue: The filename/matching queue
                        that images will be taken from.

          epoch_counter: The epoch counter.

          batch_size: (Optional). The size of the batches
                        to be generated. Defaults to
                        number listed in Flags.

          num_threads: (Optional). The number of threads
                        to be used. Defaults to number
                        listed in Flags.

    """

    # read one receptor and stack of ligands; choose one of the ligands from the stack according to epoch
    ligand_file, current_epoch, label, ligand_elements, ligand_coords, receptor_elements, receptor_coords = read_receptor_and_ligand(
        filename_queue, epoch_counter=epoch_counter, lig_frame_sampling=lig_frame_sampling)

    # convert coordinates of ligand and protein into an image
    dense_image, _, _ = complex_coords_to_image(ligand_elements, ligand_coords, receptor_elements, receptor_coords,
                                                side_pixels, pixel_size)

    # create a batch of proteins and ligands to read them together
    multithread_batch = tf.train.batch([ligand_file, current_epoch, label, dense_image], batch_size,
                                       num_threads=num_threads,
                                       capacity=batch_size * 3, dynamic_pad=True,
                                       shapes=[[], [], [], [side_pixels, side_pixels, side_pixels]])

    return multithread_batch



def read_receptor_and_ligand(filename_queue):
    """Creates a queue of images and labels based on the filename/matching queue

            Args:
              filename_queue: The filename/matching queue
                            where images and matching data
                            will be taken from

            Returns:
                A triplet containing a tensor of the key image, a tensor of the lock image,
                and a label recording whether the key and the lock match or not.

    """

    # Get id, key file, lock file, and label
    # values from the filname queue
    id_value  = filename_queue[0]
    key_file  = filename_queue[1]
    lock_file = filename_queue[2]
    label     = filename_queue[3]

    # Decode the images into tensors
    key_image = decode_image(key_file)
    lock_image = decode_image(lock_file)

    return key_image, lock_image, label



def decode_image(file_path):
    """Given an image filepath, reads in the image and returns a tensor of the image.

            Args:
              file_path: The filename of the image.


            Credits:

                This function includes code adapted from the following sources:
                    -https://stackoverflow.com/a/33862534/


    """

    # Create a small filename queue, containing just the image
    # we're going to read (This is probably quite inefficient,
    #   especially given the large number of images we're
    #   dealing with. Improvements welcome)
    filename_queue = tf.train.string_input_producer([file_path])

    # Read in the file
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)

    # Decode the image into a tensor
    image_tensor = tf.image.decode_png(value)

    return image_tensor
