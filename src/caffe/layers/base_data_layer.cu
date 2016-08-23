#include <vector>

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

template <typename Dtype, typename TBatch>
void BasePrefetchingDataLayer<Dtype, TBatch>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
      top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.gpu_data(),
        top[1]->mutable_gpu_data());
  }
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

//INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);

template void BasePrefetchingDataLayer<float, Batch<float> >::Forward_gpu( \
    const std::vector<Blob<float>*>& bottom, \
    const std::vector<Blob<float>*>& top); \
template void BasePrefetchingDataLayer<double, Batch<double> >::Forward_gpu( \
    const std::vector<Blob<double>*>& bottom, \
    const std::vector<Blob<double>*>& top);


}  // namespace caffe
