#!/bin/bash

PASCAL_DIR=data/pascal

# Download PASCAL VOC 2007 dataset.
if [ -d $PASCAL_DIR/VOCdevkit ]; then
  echo "Found PASCAL VOC data."
else
  URL=http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
  FILE=data/pascal/VOCtrainval_06-Nov-2007.tar
  if [ ! -d $PASCAL_DIR ]; then
    mkdir $PASCAL_DIR 
  fi
  echo "Downloading PASCAL VOC 2007..."
  if [ ! -f $FILE ]; then
    wget $URL -O $FILE
  fi

  echo "Unzipping..."
  tar xvf $FILE -C $PASCAL_DIR

  echo "Done."
fi


# Generate the plain text file with ground-truth.
VOCDEVKIT=data/pascal/VOCdevkit
DATASET=trainval
YEAR=2007
GT_FILE=data/pascal/${YEAR}_${DATASET}.txt
echo $GT_FILE
python examples/multibox/create_pascal_voc.py $VOCDEVKIT $YEAR $DATASET $GT_FILE

# Create a LMDB dataset.
WIDTH=224
HEIGHT=224
LMDB_DIR="data/pascal/PASCAL_VOC_2007_${WIDTH}_${HEIGHT}"

rm -r $LMDB_DIR

./build/examples/multibox/create_pascal_lmdb.bin --label_filename $GT_FILE \
  --lmdb_dir $LMDB_DIR \
  --width $WIDTH \
  --height $HEIGHT \
  --logtostderr


