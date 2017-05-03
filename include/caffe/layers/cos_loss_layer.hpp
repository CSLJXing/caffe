#ifndef CAFFE_COS_LOSS_LAYER_HPP_
#define CAFFE_COS_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class CosLossLayer : public LossLayer<Dtype> {
public:
  explicit CosLossLayer(const LayerParameter& param)
    : LossLayer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      		 const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CosLoss"; }

protected:
  /// @copydoc InnerProductLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      		     const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      		     const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      		      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      		      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> dot_;
};

}  // namespace caffe

#endif  // CAFFE_COS_LOSS_LAYER_HPP_
