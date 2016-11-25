---
title: Accuracy and Top-k
---

# Accuracy and Top-k

`Accuracy` scores the output as the accuracy of output with respect to target -- it is not actually a loss and has no backward step.

* Layer type: `Accuracy`
* Header: [`./include/caffe/layers/accuracy_layer.hpp`](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/accuracy_layer.hpp)
* CPU implementation: [`./src/caffe/layers/accuracy_layer.cpp`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/accuracy_layer.cpp)
* CUDA GPU implementation: [`./src/caffe/layers/accuracy_layer.cu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/accuracy_layer.cu)
* Parameters (`AccuracyParameter accuracy_param`)
** See [`./src/caffe/proto/caffe.proto`](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto))
