#include <vector>

#include "caffe/layers/huber_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void HuberLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_GT(this->layer_param_.huber_loss_param().delta(), 0)
    << "delta of huber loss need to be greater than 0.";
}

template <typename Dtype>
void HuberLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
    << "Inputs must have the same number of examples.";
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void HuberLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // bottom[0] is predicted values
  // bottom[1] is ground truth
  int count = bottom[0]->count();
  const Dtype* pre_data = bottom[0]->cpu_data();
  const Dtype* gt_data = bottom[1]->cpu_data();
  // For accelerating backward compute,
  // compute gradient in forward (store in diff_)
  Dtype* diff_data = diff_.mutable_cpu_data();

  caffe_sub(count, gt_data, pre_data, diff_data);

  const Dtype delta = this->layer_param_.huber_loss_param().delta();
  const Dtype delta_2 = delta * delta;
  Dtype loss = 0;
  for (int i = 0; i < count; ++i) {
    Dtype abs_diff_data = abs(diff_data[i]);
    if (abs_diff_data < delta) {
      loss += (abs_diff_data * abs_diff_data*0.5);
      diff_data[i] = -diff_data[i];
    } else {
      loss += (delta * abs_diff_data - 0.5*delta_2);
      diff_data[i] = (pre_data[i] < gt_data[i] ? -delta : delta);
    }
  }
  top[0]->mutable_cpu_data()[0] = (loss / count);
}

template <typename Dtype>
void HuberLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    int count = bottom[0]->count();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_copy(count, diff_.cpu_data(), bottom_diff);
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / count;
    caffe_scal(count, loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(HuberLossLayer);
#endif

INSTANTIATE_CLASS(HuberLossLayer);
REGISTER_LAYER_CLASS(HuberLoss);

}  // namespace caffe
