#include <vector>

#include "caffe/layers/weighted_euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void WeightedEuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";

  CHECK_EQ(bottom[2]->channels(), 1);
  CHECK_EQ(bottom[1]->num(), bottom[2]->num());
  CHECK_EQ(bottom[1]->height(), bottom[2]->height());
  CHECK_EQ(bottom[1]->width(), bottom[2]->width());

  diff_.ReshapeLike(*bottom[0]);
  l2diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void WeightedEuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());

  caffe_mul(
      count,
      diff_.cpu_data(),
      diff_.cpu_data(),
      l2diff_.mutable_cpu_data());

  Dtype loss = Dtype(0.0);
  const int outer_num = bottom[2]->count(2);
  for (int i = 0; i < bottom[0]->num(); ++i) {
    const Dtype* weights = bottom[2]->mutable_cpu_data() + i * outer_num;

    for (int j = 0; j < bottom[0]->channels(); ++j) {
      const Dtype* l2diffs = l2diff_.mutable_cpu_data() + l2diff_.offset(i, j, 0, 0);
      loss += caffe_cpu_dot(outer_num, weights, l2diffs);
    }
  }
  
  Dtype normalizer = Dtype(0.0);
  for (int i = 0; i < bottom[2]->count(); ++i) {
    normalizer += bottom[2]->cpu_data()[i];
  }
  normalizer = std::max(Dtype(1.0), normalizer);

  top[0]->mutable_cpu_data()[0] = loss / 2.0 / normalizer;
}

template <typename Dtype>
void WeightedEuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype normalizer = Dtype(0.0);
    for (int i = 0; i < bottom[2]->count(); ++i) {
      normalizer += bottom[2]->cpu_data()[i];
    }
    normalizer = std::max(Dtype(1.0), normalizer);
    caffe_scal(diff_.count(), static_cast<Dtype>(top[0]->cpu_diff()[0] / normalizer),
               diff_.mutable_cpu_data());

    const int outer_num = bottom[2]->count(2);
    for (int i = 0; i < bottom[0]->num(); ++i) {
      const Dtype* weights = bottom[2]->mutable_cpu_data() + i * outer_num;

      for (int j = 0; j < bottom[0]->channels(); ++j) {
        const Dtype* diff = diff_.cpu_data() + diff_.offset(i, j, 0, 0);
        caffe_mul(outer_num, weights, diff, 
                  bottom[0]->mutable_cpu_diff() + bottom[0]->offset(i, j, 0, 0));
      }
    }
  }
}

INSTANTIATE_CLASS(WeightedEuclideanLossLayer);
REGISTER_LAYER_CLASS(WeightedEuclideanLoss);

}  // namespace caffe
