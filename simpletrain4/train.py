import base64

from datetime import datetime
import tensorflow as tf

from simpletrain4 import st4
from utils import *



def train():
    # Make logging very verbose
    tf.logging.set_verbosity(tf.logging.DEBUG)

    notify("simpletrain4 is now running!", subject="Running!")

    # Create session
    sess = tf.Session()

    # Start coordinator and queue runners
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        print("At train step")

        # Say hi to Maksym
        if get_username() == 'maksym':
            m = "6Ojo"
            print(base64.b64decode('Ojo6ICAgIDo6OiA' + m + '' + m + '' + m + '6OiAgICAgIDo' + m + 'gICAgOjo6OiAgICAgIDo'
                                                                                                  '6OiAg'
                                                                                                  'ICAgOjo6ICAgIDo'
                                                                                                  '6OiA'
                                   + m + '' + m + '6OiAgOjo6ICAgOjo6IDo' + m + 'gICAgOjo6OiAgOjo6IA==') + "\n:+:    :+:"
                                                                                                          "    "
                                                                                                          " :+:        "
                                                                                                          "  +:+:+: :+:"
                                                                                                          "+:+   :+: :+"
                                                                                                          ":   "
                                                                                                          ":+:   :+: :+"
                                                                                                          ":    :+: :+:"
                                                                                                          "   :"
                                                                                                          "+: +:+:+: :+"
                                                                                                          ":+:+ :+: \n+"
                                                                                                          ":+  "
                                                                                                          "  +:+     +:"
                                                                                                          "+          +"
                                                                                                          ":+ +"
                                                                                                          ":+:+ +:+  +:"
                                                                                                          "+   +:+  +:+"
                                                                                                          "  +:"
                                                                                                          "+  +:+      "
                                                                                                          "   +:+ +:+  "
                                                                                                          "+:+ "
                                                                                                          "+:+:+ +:+ +:"
                                                                                                          "+ \n" +
                  base64.
                  b64decode(
                      'KyMrKzorKyMrKyAgICAgKyMrICAgICAgICAgICsjKyAgKzorICArIysgKyMrKzorKyMrKzogKyMrKzorKyAgICsjKys6Ky'
                      'sjKysgICArIysrOiAgICsjKyAgKzorICArIysgKyMrIA==') + "\n+#+    +#+     +#+          +#+       +#"
                                                                          "+ +#+     +#+ +#+  +#+         +#+    +#+ "
                                                                          "   +#+       +#+ +#+ \n#+#    #+#     #+# "
                                                                          "         #+#       #+# #+#     #+# #+#   #"
                                                                          "+# #+#    #+#    #+#    #+#       #+#     "
                                                                          "\n###    ### ###########      ###       ##"
                                                                          "# ###     ### ###    ### ########     ### "
                                                                          "   ###       ### ### ")

        # Get images and labels for MSHAPES
        images, labels = st4.inputs(eval_data=False)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = st4.inference(images)

        # Calculate loss.
        loss = st4.loss(logits, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = st4.train(loss, global_step)


        for _ in range(0, 1000):
            sess.run(train_op)




def main(argv=None):
    print("Hello, world!")

    notify("Hello, world!", subject="Hi!!!")

    missing_dependencies = check_dependencies_installed()

    if len(missing_dependencies) > 0:
        raise Exception("Not all dependencies are installed! (Missing packages " + ' and '.join(missing_dependencies) +
                        "). See README.md for details.")

    # Download and extract the dataset if it's missing
    maybe_download_and_extract()

    # Run some checks on the dataset to make sure it's correct
    verify_dataset()

    # Clean up directories
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)

    # Train the network!!
    train()



if __name__ == '__main__':
    tf.app.run()
