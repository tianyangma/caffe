#include <vector>

#include "caffe/multibox_loss_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MultiBoxLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  // Set up prior bounding boxes.
  const string &priors_file = this->layer_param_.multibox_param().priors_file();
  LOG(INFO) << "Opening prior file: " << priors_file;
  std::ifstream infile(priors_file.c_str());

  // Set up the confidence weight.
  alpha_ = this->layer_param_.multibox_param().alpha();

  vector<Dtype> priors;
  Dtype coordinate;
  int count = 0;
  while (infile >> coordinate) {
    count++;
    // Only pushes the second coordinate.
    if (count % 2 == 0) {
      priors.push_back(coordinate);
    }
  }
  CHECK(priors.size() % 4 == 0);

  const int num_priors = priors.size() / 4;
  this->prior_bounding_boxes_.clear();
  for (int i = 0; i < num_priors; ++i) {
    const int index = 4 * i;
    BoundingBox box(priors[index], priors[index + 1], priors[index + 2],
                    priors[index + 3], -1);
    this->prior_bounding_boxes_.push_back(box);
  }
}

template <typename Dtype>
void MultiBoxLossLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top) {
  // Each prediction has 5 scalars: 4 for coordinates, 1 for confidence.
  CHECK(bottom[0]->shape(1) % 5 == 0);

  // Number of predictions should be the same as priors.
  CHECK(bottom[0]->shape(1) / 5 == this->prior_bounding_boxes_.size())
      << "#predicions: " << bottom[0]->shape(1) / 5 << " "
      << "#priors: " << prior_bounding_boxes_.size();

  LossLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void MultiBoxLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                           const vector<Blob<Dtype> *> &top) {
  ground_truth_boxes_.clear();
  predicted_boxes_.clear();
  labels_.clear();

  const int batch_size = bottom[0]->shape(0);
  const int num_predictions = bottom[0]->shape(1) / 5;
  const int max_num_objects = bottom[1]->shape(1) / 5;

  Dtype *loss = top[0]->mutable_cpu_data();
  *loss = 0;
  for (int batch = 0; batch < batch_size; ++batch) {
    ground_truth_boxes_.push_back(vector<BoundingBox>());
    predicted_boxes_.push_back(vector<BoundingBox>());

    // Decode the ground-truth objects.
    const Dtype *data =
        bottom[1]->cpu_data() + bottom[1]->offset(batch, 0, 0, 0);
    for (int j = 0; j < max_num_objects; ++j) {
      const int index = 5 * j;
      BoundingBox box(data[index], data[index + 1], data[index + 2],
                      data[index + 3], -1);
      // Each image may have different number of ground-truth objects. However,
      // to have the same dimension to form a Blob, we filled -1s. Here, we
      // check if an object is a real ground-truth object.
      if (box.IsValid()) {
        ground_truth_boxes_[batch].push_back(box);
      }
    }

    // Match priors to the ground-truth objects.
    labels_.push_back(MatchPriorsToObjects(ground_truth_boxes_[batch]));
    data = bottom[0]->cpu_data() + bottom[0]->offset(batch, 0, 0, 0);
    for (int i = 0; i < num_predictions; ++i) {
      Dtype conf_loss = 0;
      Dtype loc_loss = 0;
      // Decode the prediction. Confidence is calculated by passing the 5th
      // regression value (logit) through a logistic function.
      const int index = 5 * i;
      BoundingBox predict_values(data[index], data[index + 1], data[index + 2],
                                 data[index + 3], Logistic(data[index + 4]));
      BoundingBox predict_bbox = predict_values + prior_bounding_boxes_[i];
      predicted_boxes_[batch].push_back(predict_bbox);

      const int &label = labels_[batch][i];
      if (label >= 0) {
        conf_loss = -log(predict_bbox.confidence);
        loc_loss =
            0.5 * predict_bbox.L2Distance(ground_truth_boxes_[batch][label]);
      } else {
        conf_loss = -log(1.0 - predict_bbox.confidence);
      }

      *loss += alpha_ * conf_loss;
      *loss += loc_loss;
    }
  }
}

template <typename Dtype>
void MultiBoxLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {
  const int batch_size = bottom[0]->shape(0);

  // Only propagate the gradient to predictions.
  Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
  for (int batch = 0; batch < batch_size; ++batch) {
    Dtype *diff = bottom_diff + bottom[0]->offset(batch, 0, 0, 0);
    for (int j = 0; j < predicted_boxes_[batch].size(); ++j) {
      const BoundingBox &predicted_box = predicted_boxes_[batch][j];
      const int index = 5 * j;
      const int label = labels_[batch][j];
      if (label >= 0) {
        const BoundingBox &matched_box = ground_truth_boxes_[batch][label];
        diff[index] = predicted_box.xmin - matched_box.xmin;
        diff[index + 1] = predicted_box.ymin - matched_box.ymin;
        diff[index + 2] = predicted_box.xmax - matched_box.xmax;
        diff[index + 3] = predicted_box.ymax - matched_box.ymax;
      } else {
        // Gradient will be 0 if it's a false positive.
        for (int k = 0; k < 4; ++k) {
          diff[index + k] = 0;
        }
      }
      const Dtype &c = predicted_box.confidence;

      // NOTE(tianyangm): this is in fact gradient = -1/c if true-positive, and
      // 1 / ( 1 - c) if false positive.
      diff[index + 4] = alpha_ * (c - (label >= 0)) / ((1 - c) * c);
    }
  }
}

template <typename Dtype>
vector<int> MultiBoxLossLayer<Dtype>::MatchPriorsToObjects(
    const vector<BoundingBox> &objects) const {
  // If an prior gets no match, the label will be -1.
  const int num_priors = this->prior_bounding_boxes_.size();
  vector<int> assignments(num_priors);
  for (int i = 0; i < num_priors; ++i) {
    const BoundingBox &prior = this->prior_bounding_boxes_[i];
    Dtype max_overlap = -1.0;
    int label = -1;
    for (int j = 0; j < objects.size(); ++j) {
      Dtype overlap = prior.GetOverlapRatio(objects[j]);
      if (overlap > max_overlap) {
        label = j;
        max_overlap = overlap;
      }
    }
    if (max_overlap > 0.5) {
      assignments[i] = label;
    } else {
      static const int kNoMatch = -1;
      assignments[i] = kNoMatch;
    }
  }
  return assignments;
}

template <typename Dtype>
Dtype MultiBoxLossLayer<Dtype>::Logistic(const Dtype &logit) {
  return 1.0 / (1 + exp(-logit));
}

#ifdef CPU_ONLY
STUB_GPU(MultiBoxLossLayer);
#endif

INSTANTIATE_CLASS(MultiBoxLossLayer);
REGISTER_LAYER_CLASS(MultiBoxLoss);

}  // namespace caffe
