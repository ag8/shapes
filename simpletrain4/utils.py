import importlib
import os
import pwd
import socket

import sys
import zipfile

import urllib
from random import randint

import requests

from time import gmtime, strftime

from PIL import Image
from six.moves import urllib as smurllib
from simpletrain4 import FLAGS, PARAMS



def get_username():
    return pwd.getpwuid(os.getuid()).pw_name



def check_dependencies_installed():
    """
    Checks whether the needed dependencies are installed.

    :return: a list of missing dependencies
    """
    missing_dependencies = []

    try:
        import importlib
    except ImportError:
        missing_dependencies.append("importlib")

    dependencies = ["termcolor",
                    "colorama",
                    "tensorflow",
                    "numpy",
                    "PIL",
                    "six",
                    "tarfile",
                    "zipfile",
                    "requests"]

    for dependency in dependencies:
        if not can_import(dependency):
            missing_dependencies.append(dependency)

    return missing_dependencies


def can_import(some_module):
    try:
        importlib.import_module(some_module)
    except ImportError:
        return False

    return True


def maybe_download_and_extract():
    """Downloads and extracts the zip from en, if necessary"""
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = PARAMS.DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()


        filepath, _ = smurllib.request.urlretrieve(PARAMS.DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')

    if not os.path.exists(extracted_dir_path):
        zip_ref = zipfile.ZipFile(filepath, 'r')
        zip_ref.extractall(dest_directory)
        zip_ref.close()


def verify_dataset():
    which = randint(1, 10000)

    where = os.path.join(FLAGS.data_dir, 'images/%d_L.png' % which)

    im = Image.open(where)
    width, height = im.size

    # print("w, h: " + str(width) + ", " + str(height))

    if not (width == 100 and height == 100):
        raise Exception("Dataset appears to have been corrupted. (Check " + where + ")")



def notify(message, subject="Notification", email=FLAGS.NOTIFICATION_EMAIL):
    params = {'message': "[" + get_time_string() + "]: " + message, 'subject': subject, 'email': email}
    encoded_params = urllib.urlencode(params)

    response = requests.get('https://electronneutrino.com/affinity/notify/notify.php?' + encoded_params)
    print (response.status_code)
    print (response.content)



def get_time_string():
    return strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " GMT"


def get_hostname():
    return socket.gethostname()

