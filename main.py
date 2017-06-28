import os
import time
from random import randint
import nshapegen

# Create directory for storing images
if not os.path.exists("images"):
    os.makedirs("images")


nshapegen.generate_image_pairs(50000)