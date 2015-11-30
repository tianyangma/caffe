# This script generates a file containing the training images and annotations of the PASCAL VOC dataset.
# Example usage:
# VOCDEVKIT=/data/VOCdevkit
# DATASET=trainval
# YEAR=2007
# OUTPUT=/data/2007_trainval.txt
# create_pascal_voc.py $VOCDEVKIT $YEAR $DATASET $OUTPUT

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import sys

sets = [('2007', 'train')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Normalizes the bounding box using image dimensions.
# size: (image_width, image_height)
# box: (x_min, x_max, y_min, y_max)
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x_min = box[0] * dw
    x_max = box[1] * dw
    y_min = box[2] * dh
    y_max = box[3] * dh
    return (x_min, y_min, x_max, y_max)

# Converts an XML annotation file to a string.
# The format of the string is:
# <width> <height> <num_objects> <x_min> <x_max> <y_min> <y_max> <class id> ...
def convert_annotation(annotation_filename):
    output_string = ""

    tree = ET.parse(annotation_filename)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    output_string += (" " + str(w) + " " + str(h))

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
	annotations += " " + " ".join([str(a) for a in bb])  + " " + str(cls_id)

    output_string += (" " + str(num_objects))
    output_string += (" " + annotations)
    return output_string

if __name__ == '__main__':
    voc_devkit = sys.argv[1]
    year = sys.argv[2]
    image_set = sys.argv[3]
    output_filename = sys.argv[4]

    # Set the VOC dataset path.
    voc_path = os.path.join(voc_devkit, 'VOC%s'%year)

    print 'Processing VOC%s-%s: %s' %(year, image_set, voc_path)

    # Read the image list.
    image_ids = open('%s/ImageSets/Main/%s.txt' %
                    (voc_path, image_set)).read().strip().split()
    print 'Found %d images.' % len(image_ids)

    # Parse the annotations.
    out_file = open(output_filename, 'w')
    for image_id in image_ids:
        # Make sure that the JPEG image exists.
        jpeg_filename = '%s/JPEGImages/%s.jpg' %(voc_path, image_id)
        if not os.path.exists(jpeg_filename):
            print '%s does not exist.' % jpeg_filename
            continue
        # Make sure that the annotation file exists.
        annotation_filename = '%s/Annotations/%s.xml'%(voc_path, image_id)
        if not os.path.exists(annotation_filename):
            print '%s does not exist.' % annotation_filename
            continue

        # Add one line to the annotation file.
        out_file.write(jpeg_filename)
        out_file.write(" " + convert_annotation(annotation_filename) + "\n")

    out_file.close()
