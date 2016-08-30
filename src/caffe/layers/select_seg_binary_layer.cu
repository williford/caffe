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

  CHECK_EQ(this->data_.height(), this->seg_.height())
      << "The data and segmentation label should have the same height.";
  CHECK_EQ(top[0]->height(), top[1]->height())
      << "The data and segmentation label should have the same height.";
  CHECK_EQ(top[0]->width(), top[1]->width())
      << "The data and segmentation label should have the same width.";

  //top[0]->Reshape(1, 1, 1, 1);

}

INSTANTIATE_LAYER_GPU_FORWARD(BatchSeg);

}  // namespace caffe
