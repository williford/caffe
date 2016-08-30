#ifndef CAFFE_SELECT_SEG_BINARY_LAYER_HPP_
#define CAFFE_SELECT_SEG_BINARY_LAYER_HPP_

// From: https://github.com/HyeonwooNoh/caffe (caffe/include/caffe/data_layers.hpp)

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/image_data_layer.hpp"

namespace caffe {

template <typename Dtype>
class BatchSeg : public Batch<Dtype> {
 public:
  // Batch: Blob<Dtype> data_, label_;
  Blob<Dtype> seg_;

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
};

/**
 * @brief Provides data to the Net from image and segmentation files.
 *
 */

template <typename Dtype>
class SelectSegBinaryLayer : public BasePrefetchingDataLayer<Dtype, BatchSeg<Dtype> > {
 public:
  explicit SelectSegBinaryLayer(const LayerParameter& param)
    : BasePrefetchingDataLayer<Dtype, BatchSeg<Dtype> >(param) {}
  virtual ~SelectSegBinaryLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SelectSegBinary"; }
  /* Deprecated method: virtual inline LayerParameter_LayerType type() const {
    return LayerParameter_LayerType_IMAGE_DATA;
  }*/
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 3; }
  virtual inline bool AutoTopBlobs() const { return true; }

 protected:
  //BatchSeg<Dtype> transformed_;
  Blob<Dtype> transformed_seg_;  // transformed_.seg_
  Blob<Dtype> class_label_;  // transformed_.label_

  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

  typedef struct SegItems {
    std::string imgfn;
    std::string segfn;
    int x1, y1, x2, y2;
    vector<int> cls_label;
  } SEGITEMS;

  vector<SEGITEMS> lines_;
  int lines_id_;
  int label_dim_;
};

} // namespace caffe

#endif  // CAFFE_SELECT_SEG_BINARY_LAYER_HPP_
