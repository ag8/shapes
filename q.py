from __future__ import print_function
import tensorflow as tf

f = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]
l = ["l1", "l2", "l3", "l4", "l5", "l6", "l7", "l8"]

fv = tf.constant(f)
lv = tf.constant(l)

rsq = tf.RandomShuffleQueue(10, 0, [tf.string, tf.string], shapes=[[],[]])
do_enqueues = rsq.enqueue_many([fv, lv])

gotf, gotl = rsq.dequeue()

with tf.Session() as sess:
    tf.global_variables_initializer().run(session=sess)
    tf.train.start_queue_runners(sess=sess)
    sess.run(do_enqueues)
    for i in xrange(2):
        one_f, one_l = sess.run([gotf, gotl])
        one_l = one_l + '3434'
        print("F: ", one_f, "L: ", one_l)