#include <vector>

#include "caffe/layers/select_seg_binary_layer.hpp" // for BatchSeg

namespace caffe {

template <typename Dtype>
void BatchSeg<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Reshape to loaded data.
  top[0]->ReshapeLike(this->data_);
  // Copy the data
  caffe_copy(this->data_.count(), this->data_.gpu_data(),
      top[0]->mutable_gpu_data());

  if (top.size() > 1) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(this->seg_);
    // Copy the segmentation.
    caffe_copy(this->seg_.count(), this->seg_.gpu_data(),
        top[1]->mutable_gpu_data());
  }

  if (top.size() > 2) {
    // Reshape to loaded labels.
    top[2]->ReshapeLike(this->label_);
    // Copy the labels.
    caffe_copy(this->label_.count(), this->label_.gpu_data(),
        top[2]->mutable_gpu_data());
  }

}

INSTANTIATE_LAYER_GPU_FORWARD(BatchSeg);

}  // namespace caffe
