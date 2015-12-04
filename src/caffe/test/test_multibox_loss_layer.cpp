#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/multibox_loss_layer.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class MultiBoxLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  static const Dtype kAlpha = 0.01;

protected:
  MultiBoxLossLayerTest()
      : predicts_(new Blob<Dtype>(1, 10, 1, 1)),
        ground_truths_(new Blob<Dtype>(1, 201, 1, 1)), loss_(new Blob<Dtype>()) {}

  virtual ~MultiBoxLossLayerTest() {
    delete predicts_;
    delete ground_truths_;
    delete loss_;
  }

  virtual void SetUp() {
    // On the first image, we have ground-truth as object0 (0.1, 0.1, 0.3, 0.3)
    Dtype *data = this->ground_truths_->mutable_cpu_data();

    // There're 5 labels.
    data[0] = 5;

    data[1] = 0.1;
    data[2] = 0.1;
    data[3] = 0.3;
    data[4] = 0.3;
    data[5] = 0;

    // Priors are (0.1, 0.1, 0.3, 0.3) and (0.4, 0.4, 0.6, 0.6)
    // So prior0 -> object0.
    // prior1 -> false positive.

    // Then we have prediction on the offsets.
    // predict0: (0.2, 0, 0.3, -0.1, 5)
    data = this->predicts_->mutable_cpu_data();
    *(data++) = 0.2;
    *(data++) = 0;
    *(data++) = 0.3;
    *(data++) = -0.1;
    *(data++) = -1;

    // predict1: (100, 100, 200, 200, -5);
    *(data++) = 100;
    *(data++) = 100;
    *(data++) = 200;
    *(data++) = -200;
    *(data++) = 5;

    blob_bottom_vec_.push_back(predicts_);
    blob_bottom_vec_.push_back(ground_truths_);
    blob_top_vec_.push_back(loss_);

    LayerParameter layer_param;
    MultiBoxParameter *multibox_param = layer_param.mutable_multibox_param();
    multibox_param->set_priors_file("src/caffe/test/test_data/ipriors2.txt");
    multibox_param->set_alpha(kAlpha);
    layer_.reset(new MultiBoxLossLayer<Dtype>(layer_param));
    layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  }

  void TestSetup() {
    EXPECT_EQ(this->loss_->num(), 1);
    EXPECT_EQ(this->loss_->channels(), 1);
    EXPECT_EQ(this->loss_->height(), 1);
    EXPECT_EQ(this->loss_->width(), 1);
  }

  void TestForward() {
    layer_->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // Ok, let's compute the loss manually.
    // Prior0 is a true-positive, so it contributes to both confidence
    // loss and loc loss.
    Dtype conf = 1.0 / (1.0 + exp(1));
    Dtype conf_loss0 = -log(conf); // low confidence will lead to huge loss.

    // Because the prior0 aligns perfectly with object0, so the loss is simply
    // the L2 norm of the offset.
    Dtype loc_loss0 = 0;
    loc_loss0 += 0.2 * 0.2;
    loc_loss0 += 0.3 * 0.3;
    loc_loss0 += (-0.1) * (-0.1);
    loc_loss0 *= 0.5;

    // Prior1 is a false positive, so it only contributes to confidence loss.
    conf = 1 / (1 + exp(-5));
    Dtype conf_loss1 =
        -log(1 - conf); // high confidence will lead to huge loss.

    Dtype expected_loss = kAlpha * (conf_loss0 + conf_loss1) + loc_loss0;

    const Dtype *loss = blob_top_vec_[0]->cpu_data();
    EXPECT_NEAR(*loss, expected_loss, 1e-4);
  }

  void TestBackward() {
    layer_->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // if allow propagation is hard-coded in the backward function.
    vector<bool> allow_propagation(2, false);
    layer_->Backward(this->blob_top_vec_, allow_propagation,
                     this->blob_bottom_vec_);

    // Prior0 is a true-positive, so it has non-zero gradient on location
    // predictions.
    Dtype *diff = blob_bottom_vec_[0]->mutable_cpu_diff();
    EXPECT_NEAR(diff[0], 0.2, 1e-4);
    EXPECT_NEAR(diff[1], 0, 1e-4);
    EXPECT_NEAR(diff[2], 0.3, 1e-4);
    EXPECT_NEAR(diff[3], -0.1, 1e-4);

    // The gradient on the confidence is calculated as (c - 1) / c / ( 1- c ) =
    // -1.0 / c.
    Dtype conf = 1.0 / (1.0 + exp(1));
    EXPECT_NEAR(diff[4], -1.0 * kAlpha / conf, 1e-4);

    // Prior1 is a false-positive, so it has zero gradients on location
    // predictions.
    EXPECT_NEAR(diff[5], 0, 1e-4);
    EXPECT_NEAR(diff[6], 0, 1e-4);
    EXPECT_NEAR(diff[7], 0, 1e-4);
    EXPECT_NEAR(diff[8], 0, 1e-4);

    // The gradient on the confidence is c / c / ( 1 - c ) = 1 / ( 1 - c).
    conf = 1 / (1 + exp(-5));
    EXPECT_NEAR(diff[9], 1.0 * kAlpha / (1 - conf), 1e-4);
  }

  void TestGradientCheck() {
    layer_->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // if allow propagation is hard-coded in the backward function.
    vector<bool> allow_propagation(2, false);
    layer_->Backward(this->blob_top_vec_, allow_propagation,
                     this->blob_bottom_vec_);

    Dtype loss_before = loss_->cpu_data()[0];

    // Update the bottom vector with the gradients.
    Dtype delta = -1e-5;
    Dtype *predict_data = predicts_->mutable_cpu_data();
    const Dtype *gradient = predicts_->mutable_cpu_diff();
    for (int i = 0; i < predicts_->count(0); ++i) {
      predict_data[i] += delta * gradient[i];
    }

    // Run forward again.
    layer_->Forward(blob_bottom_vec_, blob_top_vec_);
    EXPECT_GT(loss_before, loss_->cpu_data()[0]);
  }

  Blob<Dtype> *const predicts_;
  Blob<Dtype> *const ground_truths_;
  Blob<Dtype> *const loss_;

  vector<Blob<Dtype> *> blob_bottom_vec_;
  vector<Blob<Dtype> *> blob_top_vec_;

  shared_ptr<MultiBoxLossLayer<Dtype> > layer_;
};

TYPED_TEST_CASE(MultiBoxLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(MultiBoxLossLayerTest, TestSetup) { this->TestSetup(); }
TYPED_TEST(MultiBoxLossLayerTest, TestForward) { this->TestForward(); }
TYPED_TEST(MultiBoxLossLayerTest, TestBackward) { this->TestBackward(); }
TYPED_TEST(MultiBoxLossLayerTest, TestGradientCheck) {
  this->TestGradientCheck();
}

} // namespace caffe
