#ifndef CAFFE_REVERSE_LAYER_HPP_
#define CAFFE_REVERSE_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/*
 * Note: implementation of tf.reverse
 * https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/reverse
 */

template <typename Dtype> class ReverseLayer : public Layer<Dtype> {
public:
  explicit ReverseLayer(const LayerParameter &param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);
  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top);

  virtual inline const char *type() const { return "Reverse"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);
  /// @brief Not implemented
  virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                            const vector<bool> &propagate_down,
                            const vector<Blob<Dtype> *> &bottom) {
    NOT_IMPLEMENTED;
  }
  inline vector<int> indices(int offset, const vector<int> &shape) const;
  inline int offset(const vector<Blob<Dtype> *> &bottom,
                    const vector<int> &axis_ind,
                    const vector<int> &indices) const;
  inline vector<int> reverse_indices(const vector<int> &indices,
                                     const vector<int> &shape) const;

  vector<int> axis_;
};

} // namespace caffe

#endif // CAFFE_REVERSE_LAYER_HPP_
