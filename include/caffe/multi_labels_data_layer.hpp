#ifndef CAFFE_MULTI_LABELS_DATA_LAYER_HPP_
#define CAFFE_MULTI_LABELS_DATA_LAYER_HPP_

#include "caffe/common.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
class MultiLabelsDataLayer : public BasePrefetchingDataLayer<Dtype> {
public:
  explicit MultiLabelsDataLayer(const LayerParameter &param);
  virtual ~MultiLabelsDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype> *> &bottom,
                              const vector<Blob<Dtype> *> &top);

  // DataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char *type() const { return "MultiLabelsData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

protected:
  virtual void load_batch(Batch<Dtype> *batch);

  DataReader reader_;
};

} // namespace caffe

#endif // CAFFE_MULTI_LABELS_DATA_LAYER_HPP_
