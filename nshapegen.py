from PIL import Image, ImageDraw
from random import randint
import numpy as np
import math

import nshapegenflags

DIM = nshapegenflags.DIM  # Shorter
COLOR = nshapegenflags.COLOR


def set_dim(dim):
    nshapegenflags.DIM = dim


def set_color(color):
    nshapegenflags.COLOR = color


def generate_image_pairs(n):
    for i in range (1, n + 1):
        if (i % 100 == 99):
            print str(i + 1) + " / " + str(n)
        save_image_pair(randint(0, 2), i)


def save_image_pair(shape, id):
    (top, bottom) = get_image_pair(shape)

    bottom.save("images/" + str(id) + "_L.png")
    top.save("images/" + str(id) + "_K.png")


def get_image_pair(shape):
    im = get_shape_image(shape)

    top = im.crop((0, 0, DIM, DIM / 2))
    top.load()

    bottom = im.crop((0, DIM / 2, DIM, DIM))
    bottom.load()

    return (top, bottom)



def get_shape_image(shape):
    im = Image.new('RGB', (DIM, DIM))

    if shape == 0:
        im = ellipse(im)
    elif shape == 1:
        im = triangle(im)
    elif shape == 2:
        im = square(im)


    return im


def square(im, size=0.6 * DIM):
    # Square in the center of the image
    vertices = [(DIM / 2 - size / 2, DIM / 2 - size / 2), (DIM / 2 - size / 2, DIM / 2 + size / 2),
                (DIM / 2 + size / 2, DIM / 2 + size / 2), (DIM / 2 + size / 2, DIM / 2 - size / 2)]

    # vertices = rotate(vertices, random_angle())

    # Draw it on the image
    draw = ImageDraw.Draw(im)
    draw.polygon(vertices, fill=COLOR, outline=COLOR)
    del draw

    im = im.rotate(random_angle_degrees())

    return im


def triangle(im, size=0.6 * DIM):
    # Triangle in the center of the image
    vertices = [(DIM / 2 - size / 2, DIM / 2 + size * math.sqrt(3) / 4),
                (DIM / 2 + size / 2, DIM / 2 + size * math.sqrt(3) / 4), (DIM / 2, DIM / 2 - size * math.sqrt(3) / 4)]

    # vertices = rotate(vertices, random_angle())

    # Draw it on the image
    draw = ImageDraw.Draw(im)
    draw.polygon(vertices, fill=COLOR, outline=COLOR)
    del draw

    im = im.rotate(random_angle_degrees())

    return im


def ellipse(im):
    # Generate random bounds of ellipse; avoid having ellipses that are too narrow
    e_x = randint(math.floor(0.2 * DIM), math.floor(0.8 * DIM))
    e_y = randint(DIM - e_x, 0.8 * DIM)

    # Draw ellipse
    bbox = (DIM / 2 - e_x / 2, DIM / 2 - e_y / 2, DIM / 2 + e_x / 2, DIM / 2 + e_y / 2)
    draw = ImageDraw.Draw(im)
    draw.ellipse(bbox, fill=COLOR, outline=COLOR)
    del draw

    im = im.rotate(random_angle_degrees())

    return im


def rotate(points, angle):
    result = list()

    # Generate rotation matrix based on angle
    rotation_matrix = np.matrix([[math.cos(angle), 0 - math.sin(angle)], [math.sin(angle), math.cos(angle)]])

    # Loop through every point in the list of vertices
    for point in points:
        # Create a vector out of the point: (while setting center to origin)
        #   [[x]
        #    [y]]
        point_vector = np.transpose(np.matrix([point[0] - DIM / 2, point[1] - DIM / 2]))

        # Multiply the rotation matrix by the point vector
        rotated = np.matmul(rotation_matrix, point_vector)

        # Reshape matrix into coordinate pair (and move origin back to origin from center)
        rotated_vertices = (math.floor(rotated.item((0, 0))) + DIM / 2, math.floor(rotated.item((1, 0))) + DIM / 2)

        # Add rotated vertex to list
        result.append(rotated_vertices)

    return result


def random_angle():
    return randint(0, 359) * math.pi / 180  # randint is inclusive, so 0 to 359


def random_angle_degrees():
    return randint(0, 359)
