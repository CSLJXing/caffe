#include <algorithm>
#include <vector>

#include "caffe/layers/huber_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void HuberForward(const int n, Dtype* loss, Dtype* diff,
    const Dtype* pre_data, const Dtype* gt_data, const Dtype delta) {
  Dtype delta_2 = delta * delta;
  CUDA_KERNEL_LOOP(index, n) {
    Dtype abs_diff = abs(diff[index]);
    if (abs_diff < delta) {
      loss[index] = (0.5 * abs_diff * abs_diff);
      diff[index] = -diff[index];
    } else {
      loss[index] = (delta * abs_diff - 0.5*delta_2);
      diff[index] = (pre_data[index] < gt_data[index] ? -delta : delta);
    }
  }
}

template <typename Dtype>
void HuberLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // bottom[0] is predicted values
  // bottom[1] is ground truth
  int count = bottom[0]->count();
  const Dtype* pre_data = bottom[0]->gpu_data();
  const Dtype* gt_data = bottom[1]->gpu_data();
  // For accelerating backward compute,
  // compute gradient in forward (store in diff_)
  Dtype* diff_data = diff_.mutable_gpu_data();

  caffe_gpu_sub(count, gt_data, pre_data, diff_data);

  const Dtype delta = this->layer_param_.huber_loss_param().delta();

  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();

  // NOLINT_NEXT_LINE(whitespace/operators)
  HuberForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, loss_data, diff_data, pre_data, gt_data, delta);
  CUDA_POST_KERNEL_CHECK;

  Dtype loss;
  caffe_gpu_asum(count, loss_data, &loss);
  top[0]->mutable_cpu_data()[0] = (loss / count);
}

template <typename Dtype>
void HuberLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    int count = bottom[0]->count();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(count, diff_.gpu_data(), bottom_diff);
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / count;
    caffe_gpu_scal(count, loss_weight, bottom_diff);
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(HuberLossLayer);
}  // namespace caffe
