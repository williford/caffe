#ifndef CAFFE_CLASS_NORMALIZED_LOSS_LAYERS_HPP_
#define CAFFE_CLASS_NORMALIZED_LOSS_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


/**
 * @brief Computes the multinomial logistic loss for a one-of-many
 *        classification task, normalizing the number of instances of each
 *        class within each example, passing real-valued predictions through a
 *        softmax to get a probability distribution over classes.
 */
template <typename Dtype>
class SoftmaxWithClassNormalizedLossLayer : public LossLayer<Dtype> {
 public:
  explicit SoftmaxWithClassNormalizedLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param),
        softmax_layer_(new SoftmaxLayer<Dtype>(param)) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const {
    return "SoftmaxWithClassNormalizedLoss";
  }
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// The internal SoftmaxLayer used to map predictions to a distribution.
  shared_ptr<SoftmaxLayer<Dtype> > softmax_layer_;
  /// prob stores the output probability predictions from the SoftmaxLayer.
  Blob<Dtype> prob_;
  /// bottom vector holder used in call to the underlying SoftmaxLayer::Forward
  vector<Blob<Dtype>*> softmax_bottom_vec_;
  /// top vector holder used in call to the underlying SoftmaxLayer::Forward
  vector<Blob<Dtype>*> softmax_top_vec_;
  /// Whether to ignore instances with a certain label.
  bool has_ignore_label_;
  /// The label indicating that an instance should be ignored.
  int ignore_label_;
  /// The number of classes
  int num_classes_;
  /// Whether to normalize the loss by the total number of values present
  /// (otherwise just by the batch size).
  bool normalize_;
};


}

#endif  // CAFFE_CLASS_NORMALIZED_LOSS_LAYERS_HPP_
