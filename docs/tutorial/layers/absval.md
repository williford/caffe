---
title: Absolute Value Layer
---

# Absolute Value Layer

* Layer type: `AbsVal`
* CPU implementation: `./src/caffe/layers/absval_layer.cpp`
* CUDA GPU implementation: `./src/caffe/layers/absval_layer.cu`
* Sample

      layer {
        name: "layer"
        bottom: "in"
        top: "out"
        type: "AbsVal"
      }

The `AbsVal` layer computes the output as abs(x) for each input element x.
