#include <vector>

#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/select_seg_binary_layer.hpp" // for BatchSeg

namespace caffe {

template <typename Dtype>
void Batch<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Reshape to loaded data.
  top[0]->ReshapeLike(data_);
  // Copy the data
  caffe_copy(data_.count(), data_.gpu_data(),
      top[0]->mutable_gpu_data());
  if (top.size() > 1) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(label_);
    // Copy the labels.
    caffe_copy(label_.count(), label_.gpu_data(),
        top[1]->mutable_gpu_data());
  }
}


template <typename Dtype, typename TBatch>
void BasePrefetchingDataLayer<Dtype, TBatch>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");

  batch->Forward_gpu(bottom, top);

  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(Batch);
//INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);

template void BasePrefetchingDataLayer<float, Batch<float> >::Forward_gpu( \
    const std::vector<Blob<float>*>& bottom, \
    const std::vector<Blob<float>*>& top); \
template void BasePrefetchingDataLayer<double, Batch<double> >::Forward_gpu( \
    const std::vector<Blob<double>*>& bottom, \
    const std::vector<Blob<double>*>& top);

template void BasePrefetchingDataLayer<float, BatchSeg<float> >::Forward_gpu( \
    const std::vector<Blob<float>*>& bottom, \
    const std::vector<Blob<float>*>& top); \
template void BasePrefetchingDataLayer<double, BatchSeg<double> >::Forward_gpu( \
    const std::vector<Blob<double>*>& bottom, \
    const std::vector<Blob<double>*>& top);

}  // namespace caffe
