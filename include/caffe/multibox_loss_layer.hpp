#ifndef CAFFE_MULTIBOX_LOSS_LAYER_HPP_
#define CAFFE_MULTIBOX_LOSS_LAYER_HPP_

#include "caffe/loss_layers.hpp"

namespace caffe {

template <typename Dtype> 
class MultiBoxLossLayer : public LossLayer<Dtype> {
public:
  explicit MultiBoxLossLayer(const LayerParameter &param)
      : LossLayer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);

  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top);

  virtual inline const char *type() const { return "MultiBoxLoss"; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);
  virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                            const vector<bool> &propagate_down,
                            const vector<Blob<Dtype> *> &bottom);
private:
  class BoundingBox {
  public:
    BoundingBox() { xmin = ymin = xmax = ymax = confidence = -1; }
    BoundingBox(Dtype xmin, Dtype ymin, Dtype xmax, Dtype ymax,
                Dtype confidence)
        : xmin(xmin), ymin(ymin), xmax(xmax), ymax(ymax),
          confidence(confidence) {}
    ~BoundingBox() {}

    // Returns true if at least one coordinate is larger or equal than 0.
    bool IsValid() {
      if (xmin < 0 && ymin < 0 && xmax < 0 && ymax < 0) {
        return false;
      }
      return true;
    }

    // Returns the area.
    Dtype area() const {
      Dtype width = xmax - xmin;
      Dtype height = ymax - ymin;
      return width * height;
    }

    // Returns the overlap ratio to another BoundingBox. Overlap ratio is
    // defined as intersection area / union area.
    Dtype GetOverlapRatio(const BoundingBox &box) const {
      Dtype x1 = std::max(xmin, box.xmin);
      Dtype y1 = std::max(ymin, box.ymin);
      Dtype x2 = std::min(xmax, box.xmax);
      Dtype y2 = std::min(ymax, box.ymax);
      Dtype overlap = BoundingBox(x1, y1, x2, y2, -1).area();
      if (overlap <= 0) {
        return 0;
      } else {
        return overlap / (area() + box.area() - overlap);
      }
    }

    BoundingBox operator+(const BoundingBox &other) const {
      Dtype x1 = xmin + other.xmin;
      Dtype y1 = ymin + other.ymin;
      Dtype x2 = xmax + other.xmax;
      Dtype y2 = ymax + other.ymax;
      return BoundingBox(x1, y1, x2, y2, confidence);
    }

    Dtype L2Distance(const BoundingBox &other) const {
      Dtype dist = 0.0;
      dist += (xmin - other.xmin) * (xmin - other.xmin);
      dist += (ymin - other.ymin) * (ymin - other.ymin);
      dist += (xmax - other.xmax) * (xmax - other.xmax);
      dist += (ymax - other.ymax) * (ymax - other.ymax);
      return dist;
    }

    Dtype xmin;
    Dtype ymin;
    Dtype xmax;
    Dtype ymax;
    Dtype confidence;
  };

  // Matches the prior bounding boxes to a given object list and returns the
  // assignments.
  vector<int> MatchPriorsToObjects(const vector<BoundingBox> &objects) const;

  // Returns the probablity by passing the real value through a logistic
  // function.
  static Dtype Logistic(const Dtype &logit);

  vector<BoundingBox> prior_bounding_boxes_;
  vector<vector<BoundingBox> > ground_truth_boxes_;
  vector<vector<BoundingBox> > predicted_boxes_;
  vector<vector<int> > labels_;

  // The weight on the confidence loss.
  Dtype alpha_;
};

} // namespace caffe

#endif // CAFFE_MULTIBOX_LOSS_LAYER_HPP_
