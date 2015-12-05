#! /usr/bin/python
# usage: python display_pascal_lmdb.py [-h] -d DB_PATH

import caffe, lmdb, cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--db_path", type=str, dest="db_path", help="specify your voc lmdb path", required=True)
args = parser.parse_args()

db_path = args.db_path

env = lmdb.open(db_path, readonly=True)
with env.begin() as txn:
    cursor = txn.cursor()
    for filename, raw_datum in cursor:
        print filename

        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(raw_datum)
        print 'float data: ', datum.float_data

        x = caffe.io.datum_to_array(datum)
        x = x.swapaxes(0,1).swapaxes(1,2)

        cv2.imshow('img', x)
        c = cv2.waitKey()
        if c==27:
            break
