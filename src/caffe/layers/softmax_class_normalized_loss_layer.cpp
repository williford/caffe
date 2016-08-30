#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/softmax_class_normalized_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithClassNormalizedLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  normalize_ = this->layer_param_.loss_param().normalize();
  if (normalize_!=true) {
    LOG(FATAL) << this->type()
               << " Layer cannot have normalization set to false.";
  }

  // hard-code, since I don't expect to use this code for more than 2 classes
  num_classes_ = 2; 
}

template <typename Dtype>
void SoftmaxWithClassNormalizedLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void SoftmaxWithClassNormalizedLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  if (num_classes_ != bottom[0]->channels()) {
    LOG(FATAL) << this->type()
               << " Bottom layer must have exactly " << num_classes_ << " layers.";
  }
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int num = prob_.num();  // number of examples in batch
  int dim = prob_.count() / num;  // H * W * Channels
  int spatial_dim = prob_.height() * prob_.width();
  Dtype total_loss = 0;
  for (int i = 0; i < num; ++i) {  // iterate over batch
    Dtype i_loss = 0;

    // calculate the number of positions for given class label k
    int count[2];
    for (int k = 0; k < num_classes_; k++) {
      count[k] = 0;
      for (int j = 0; j < spatial_dim; j++) { // iterate over positions
        const int label_value = static_cast<int>(label[i * spatial_dim + j]);
        if (label_value == k) {
          ++count[k];
        }
      }
    }

    for (int j = 0; j < spatial_dim; j++) { // iterate over positions
      const int label_value = static_cast<int>(label[i * spatial_dim + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.channels());
      DCHECK_GT( count[label_value], 0 );  // should be impossible
      i_loss -= (
        log(std::max(prob_data[i * dim + label_value * spatial_dim + j],
                          Dtype(FLT_MIN)))
         / count[label_value] );
    }
    total_loss += i_loss / num;
  }

  top[0]->mutable_cpu_data()[0] = total_loss;

  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithClassNormalizedLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (num_classes_ != bottom[0]->channels()) {
    LOG(FATAL) << this->type()
               << " Bottom layer must have exactly " << num_classes_ << " layers.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();

    //caffe_copy(prob_.count(), prob_data, bottom_diff); // init bottom_diff

    const Dtype* label = bottom[1]->cpu_data();
    int num = prob_.num();
    int dim = prob_.count() / num;
    int spatial_dim = prob_.height() * prob_.width();


    for (int i = 0; i < num; ++i) { // iterate over the batch

      // calculate the number of positions for given class label k
      int count[2];
      for (int c = 0; c < num_classes_; c++) {
        count[c] = 0;
        for (int j = 0; j < spatial_dim; j++) { // iterate over positions
          const int label_value = static_cast<int>(label[i * spatial_dim + j]);
          if (label_value == c) {
            ++count[c];
          }
        }
      }

      // now need to calculate gradients
      for (int j = 0; j < spatial_dim; ++j) {
        const int label_value = static_cast<int>(label[i * spatial_dim + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          for (int c = 0; c < bottom[0]->channels(); ++c) {
            bottom_diff[i * dim + c * spatial_dim + j] = 0;
          }
        } else {
          for (int c = 0; c < num_classes_; c++) {
            if (label_value == c) {
              bottom_diff[i * dim + c * spatial_dim + j] = 
                (prob_data[i * dim + c * spatial_dim + j] - 1) / count[label_value];
            } else {
              bottom_diff[i * dim + c * spatial_dim + j] = 
                prob_data[i * dim + c * spatial_dim + j] / count[label_value];
            }
          }
        }
      }
    }
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(prob_.count(), loss_weight / num, bottom_diff);
  }
}

INSTANTIATE_CLASS(SoftmaxWithClassNormalizedLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithClassNormalizedLoss);

}  // namespace caffe
