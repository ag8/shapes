import os

import nshapegen
from shape_generation import nshapegenflags


# Create directory for storing images
if not os.path.exists("images"):
    os.makedirs("images")


nshapegen.generate_image_pairs(nshapegenflags.IMAGE_NUM)
