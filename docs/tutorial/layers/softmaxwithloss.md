---
title: Softmax with Loss Layer
---

# Softmax with Loss Layer

* Layer type: `SoftmaxWithLoss`

The softmax loss layer computes the multinomial logistic loss of the softmax of its inputs. It's conceptually identical to a softmax layer followed by a multinomial logistic loss layer, but provides a more numerically stable gradient.
