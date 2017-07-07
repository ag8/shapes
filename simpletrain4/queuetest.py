import tensorflow as tf


first_list = ["a", "b", "c", "d", "e", "f", "g"]
second_list = ["1", "2", "3", "4", "5", "6", "7"]

letters = tf.convert_to_tensor(first_list, dtype=tf.string)
numbers = tf.convert_to_tensor(second_list, dtype=tf.string)

queue = tf.train.slice_input_producer([letters, numbers],
                                      num_epochs=None, shuffle=True)

key_file = queue[0]
lock_file = queue[1]

key_file2 = queue[0]
lock_file2 = queue[1]


with tf.Session() as sess:
    tf.global_variables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    k, l, k2, l2 = sess.run([key_file, lock_file, key_file2, lock_file2])

    print("k : " + k)
    print("l : " + l)
    print("k2: " + k2)
    print("l2: " + l2)
