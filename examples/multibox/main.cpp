// A sample program that uses the ImageLocalizationLayer to read object label
// and bounding boxes.
// TODO(tianyang): I am being lazy. Will write tests!

#include <vector>

#include <glog/logging.h>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/vision_layers.hpp"

using namespace caffe;
using namespace std;

#define Dtype float

DEFINE_string(root_folder, "", "Root folder of the PASCAL devkit.");
DEFINE_string(source, "", "Filename of the PASCAL ground-truths.");

int main(int argc, char **argv) {
  FLAGS_alsologtostderr = 1;
  caffe::GlobalInit(&argc, &argv);

  CHECK(!FLAGS_source.empty())
      << "You have to set the filename of the PASCAL ground-truths.";

  vector<Blob<Dtype> *> blob_bottom_vec;
  vector<Blob<Dtype> *> blob_top_vec;

  blob_top_vec.push_back(new Blob<Dtype>());
  blob_top_vec.push_back(new Blob<Dtype>());

  LayerParameter param;
  ImageDataParameter *image_data_param = param.mutable_image_data_param();
  image_data_param->set_root_folder(FLAGS_root_folder);
  image_data_param->set_source(FLAGS_source);
  static const int kBatchSize = 3;
  static const int kWidth = 100;
  static const int kHeight = 100;
  image_data_param->set_batch_size(kBatchSize);
  image_data_param->set_shuffle(false);
  image_data_param->set_new_height(kHeight);
  image_data_param->set_new_width(kWidth);

  ImageLocalizationDataLayer<Dtype> layer(param);
  layer.SetUp(blob_bottom_vec, blob_top_vec);
  layer.Forward(blob_bottom_vec, blob_top_vec);

  const Blob<Dtype>* blob = blob_top_vec[1];
  for (int i = 0; i < blob->count(0); ++i) {
    LOG(INFO) << blob->cpu_data()[i];
  }

  // Check the size of the labels.
  CHECK_EQ(blob_top_vec[1]->num(), kBatchSize) << "Incorrect batch size.";
  CHECK_EQ(blob_top_vec[1]->channels() % 5, 0) << "Incorrect number of labels.";

  // Check the size of the output image.
  CHECK_EQ(blob_top_vec[0]->num(), kBatchSize) << "Incorrect batch size.";
  CHECK_EQ(blob_top_vec[0]->channels(), 3) << "Not RGB image.";
  CHECK_EQ(blob_top_vec[0]->width(), kWidth) << "Width is incorrect";
  CHECK_EQ(blob_top_vec[0]->height(), kHeight) << "Height is incorrect.";

  LOG(INFO) << "Setup success.";
  return 0;
}
