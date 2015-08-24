# This script outputs a file that containing the image name, class labels and bounding boxes.
# Execute this script from the same directory that contains VOCdevkit.

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets = [('2007', 'train')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Normalizes the bounding box using image dimensions.


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

# Parses the annotation of an image and writes the output by appending one line to the output file.
# The format of the line is:
# <image filename> <width> <height> <num_objects> <class id> <center x> <center y> <width> <height> ...
# NOTE(tianyang):
# image filename is a relative path. So we don't need to re-generate the annotations file when moving files around.
# Root path can be specified in the image data layer params.


def convert_annotation(year, image_id, out_file):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml' % (year, image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    # The first token is the image filename.
    out_file.write('VOCdevkit/VOC%s/JPEGImages/%s.jpg' % (year, image_id))

    # Count how many objects are present in the image.
    num_objects = 0
    annotations = ""

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
    	num_objects += 1
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(
            xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
	annotations += " " + str(cls_id) + " " + " ".join([str(a) for a in bb])

    out_file.write(" " + str(w) + " " + str(h))
    out_file.write(" " + str(num_objects))
    out_file.write(annotations)
    out_file.write("\n")

wd = getcwd()

for year, image_set in sets:
    if not os.path.exists('VOCdevkit/VOC%s/labels/' % (year)):
        os.makedirs('VOCdevkit/VOC%s/labels/' % (year))
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt' %
                     (year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt' % (year, image_set), 'w')
    out_file = open('VOCdevkit/VOC%s/label.txt' % year, 'w')
    for image_id in image_ids:
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n' %
                        (wd, year, image_id))
        convert_annotation(year, image_id, out_file)
    list_file.close()
