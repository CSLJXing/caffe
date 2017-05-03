#include <vector>

#include "caffe/layers/cos_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CosLossLayer<Dtype>::Forward_gpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int num = bottom[0]->num();
  int d = bottom[0]->count(1);
  int offset = 0;
  Dtype dot = 0;
  Dtype* dot_data = this->dot_.mutable_cpu_data();
  const Dtype* bottom_0_data = bottom[0]->gpu_data();
  const Dtype* bottom_1_data = bottom[1]->gpu_data();
  for (int idx = 0; idx < num; ++idx) {
    offset = bottom[0]->offset(idx);
    caffe_gpu_dot(d, bottom_0_data+offset, bottom_1_data+offset, dot_data+idx);
    dot += dot_data[idx];
  }
  Dtype loss = 1 - dot / bottom[0]->num();
  top[0]->mutable_cpu_data()[0] = loss;

  // int count = bottom[0]->count();
  // Dtype dot;
  // caffe_gpu_dot(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), &dot);
  // Dtype loss = dot / bottom[0]->num();
  // top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void CosLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* dot_data = this->dot_.cpu_data();
    const Dtype* bottom_0_data = bottom[0]->gpu_data();
    Dtype* bottom_0_diff = bottom[0]->mutable_gpu_diff();
    int num = bottom[0]->num();
    int d = bottom[0]->count(1);
    int offset = 0;
    const Dtype beta = -Dtype(1) / num;
    caffe_copy(bottom[0]->count(),
	       bottom[1]->gpu_data(), bottom_0_diff);
    for (int idx = 0; idx < num; ++idx) {
      offset = bottom[0]->offset(idx);
      const Dtype alpha = dot_data[idx] / num;
      caffe_gpu_axpby(
	d,
	alpha,
	bottom_0_data+offset,
	beta,
	bottom_0_diff+offset);
    }
  }

  // if (propagate_down[0]) {
  //   const Dtype alpha = -top[0]->cpu_diff()[0] / bottom[0]->num();
  //   caffe_gpu_axpby(
  //     bottom[0]->count(),
  //     alpha,
  //     bottom[1]->gpu_data(),
  //     Dtype(0),
  //     bottom[0]->mutable_gpu_diff());
  // }
}

INSTANTIATE_LAYER_GPU_FUNCS(CosLossLayer);

}  // namespace caffe
