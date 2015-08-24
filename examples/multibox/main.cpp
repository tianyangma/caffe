#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/vision_layers.hpp"

using namespace caffe;
using namespace std;

#define Dtype float

int main(int argc, char** argv) {
	vector<Blob<Dtype>*> blob_bottom_vec;
	vector<Blob<Dtype>*> blob_top_vec;

	blob_top_vec.push_back(new Blob<Dtype>());
    blob_top_vec.push_back(new Blob<Dtype>());


	LayerParameter param;
  	ImageDataParameter* image_data_param = param.mutable_image_data_param();
  	image_data_param->set_batch_size(5);
  	image_data_param->set_root_folder("/home/tianyang/Downloads/");
  	image_data_param->set_source("/home/tianyang/Downloads/VOCdevkit/VOC2007/label.txt");
  	image_data_param->set_shuffle(false);
  	image_data_param->set_new_height(100);
  	image_data_param->set_new_width(100);

  	ImageLocalizationDataLayer<Dtype> layer(param);
  	layer.SetUp(blob_bottom_vec, blob_top_vec);
  	layer.Forward(blob_bottom_vec, blob_top_vec);
  	const Dtype* label = blob_top_vec[1]->cpu_data();
  	for(int i = 0; i < blob_top_vec[1]->count(); ++i) {
  		LOG(INFO) << label[i];
  	}

  	LOG(INFO) << "Setup success.";
	return 0;
}
