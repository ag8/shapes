import tensorflow as tf


# queue = tf.FIFOQueue(capacity=100, dtypes=[tf.string, tf.string], shapes=[7, 7])
queue = tf.FIFOQueue(capacity=100, dtypes=[tf.string, tf.string])

a = queue.enqueue(["a", "1"])
b = queue.enqueue(["b", "2"])
queue.enqueue(["c", "3"])
queue.enqueue(["d", "4"])
queue.enqueue(["e", "5"])
queue.enqueue(["f", "6"])
queue.enqueue(["g", "7"])
queue.enqueue(["h", "8"])



k = queue.dequeue()
l = queue.dequeue()


with tf.Session() as sess:
    tf.global_variables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    for i in xrange(1, 3):
        sess.run(a)
        sess.run(b)
        print("aaa")

        print(sess.run([k, l]))

        # print(mk + " " + ml)
