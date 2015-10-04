#!/bin/bash

if [ -f examples/multibox/VGG_ILSVRC_16_layers.caffemodel ];
then
  echo "Model file exists."
else
  wget -O examples/multibox/VGG_ILSVRC_16_layers.caffemodel \
    http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
fi

./build/tools/caffe train \
  --solver examples/multibox/solver.prototxt \
  --weights examples/multibox/VGG_ILSVRC_16_layers.caffemodel \
  --logtostderr
