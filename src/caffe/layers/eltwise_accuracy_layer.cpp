#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/eltwise_accuracy_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EltwiseAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.eltwise_accuracy_param().top_k();

  has_ignore_label_ =
    this->layer_param_.eltwise_accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.eltwise_accuracy_param().ignore_label();
  }

  has_use_label_ =
    this->layer_param_.eltwise_accuracy_param().has_use_label();
  if (has_use_label_) {
    use_label_ = this->layer_param_.eltwise_accuracy_param().use_label();
  }
}

template <typename Dtype>
void EltwiseAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0))
      << "The data and label should have the same number of examples per batch.";
  CHECK_LE(top_k_, bottom[0]->count() / bottom[0]->shape(0))
      << "top_k must be less than or equal to the number of classes.";
  CHECK_EQ(bottom[1]->shape(1), 1)
      << "Label data should have 1 channel.";
  CHECK_EQ(bottom[0]->shape(2), bottom[1]->shape(2))
      << "The data and label should have the same height.";
  CHECK_EQ(bottom[0]->shape(3), bottom[1]->shape(3))
      << "the data and label should have the same width.";
  top[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void EltwiseAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype accuracy = 0;
  // bottom_data has a separate channel for each class that is a heat map for that class
  // such as foreground / background class
  const Dtype* bottom_data = bottom[0]->cpu_data();
  // bottom_label is a single channel segmentation map (ground truth)
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->shape(0);  // num of example in each batch
  int dim = bottom[0]->count() / bottom[0]->shape(0);
  int spatial_dim = bottom[0]->shape(2) * bottom[0]->shape(3);
  int channels = bottom[0]->shape(1);
  int ignored_pixel_num = 0;
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; j++){
      const int label_value = static_cast<int>(bottom_label[i * spatial_dim + j]);
      if (has_use_label_ && label_value != use_label_) {
        ignored_pixel_num++;
        continue;
      }
      if (has_ignore_label_ && label_value == ignore_label_) {
        ignored_pixel_num++;
        continue;
      }
      /* The following finds which channels in bottom_data have the highest
       * activity. If one of the top-K channels matches the true label, then it
       * is counted as a match.
       */
      std::vector<std::pair<Dtype, int> > bottom_data_vector;
      for (int k = 0; k < channels; ++k) {
        bottom_data_vector.push_back(
          std::make_pair(bottom_data[i * dim + k * spatial_dim + j], k));
      }
      // sorts by first element in pair, and then second element
      std::partial_sort(
          bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
          bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
      // check if true label is in top k predictions
      for (int k = 0; k < top_k_; k++) {
        if (bottom_data_vector[k].second == label_value) {
          ++accuracy;
          break;
        }
      }
    }
  }
  // LOG(INFO) << "EltwiseAccuracy: " << eltwise_accuracy;
  top[0]->mutable_cpu_data()[0] = accuracy / (num * spatial_dim - ignored_pixel_num);
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(EltwiseAccuracyLayer);
REGISTER_LAYER_CLASS(EltwiseAccuracy);
}  // namespace caffe
