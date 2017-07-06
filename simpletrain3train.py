from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime
import time

from colorama import init
from colorama import Fore, Back, Style
from termcolor import colored

import pwd
import tensorflow as tf

import simpletrain3


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/MSHAPEStrain',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")



def train():
    # Make logging very verbose
    tf.logging.set_verbosity(tf.logging.DEBUG)


    from simpletrain4.utils import notify
    notify("Running!")


    sess = tf.Session()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        print("At train step!!")

        # Say hi to Maksym
        if get_username() == 'maksym':
            print("Hi Maksym!")

        # Get images and labels for MSHAPES
        images, labels = simpletrain3.inputs(eval_data=False)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = simpletrain3.inference(images)

        # Calculate loss.
        loss = simpletrain3.loss(logits, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = simpletrain3.train(loss, global_step)


        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""


            def begin(self):
                self._step = -1
                self._start_time = time.time()


            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.


            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))


        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.train_dir,
                hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                       tf.train.NanTensorHook(loss),
                       _LoggerHook()],
                config=tf.ConfigProto(
                    log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)

                # for _ in range(1000):
                #     sess.run(train_op)



def main(argv=None): #TODO: Verify the data set!!!
    init()
    print('\x1b[6;30;42m' + 'Success!' + '\x1b[0m')
    print(colored('hello', 'red'))
    simpletrain3.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()



def get_username():
    return pwd.getpwuid(os.getuid()).pw_name



if __name__ == '__main__':
    tf.app.run()
