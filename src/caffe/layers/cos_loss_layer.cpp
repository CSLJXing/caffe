#include <vector>

#include "caffe/layers/cos_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CosLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
    << "Inputs must have the same number of example";
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
    << "Inputs must have the same dimension.";
  vector<int> dot_shape(1, bottom[0]->num());
  this->dot_.Reshape(dot_shape);
}

template <typename Dtype>
void CosLossLayer<Dtype>::Forward_cpu(
   const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int num = bottom[0]->num();
  int d = bottom[0]->count(1);
  int offset = 0;
  Dtype dot = 0;
  Dtype* dot_data = this->dot_.mutable_cpu_data();
  const Dtype* bottom_0_data = bottom[0]->cpu_data();
  const Dtype* bottom_1_data = bottom[1]->cpu_data();
  for (int idx = 0; idx < num; ++idx) {
    offset = bottom[0]->offset(idx);
    dot_data[idx] = caffe_cpu_dot(d, bottom_0_data+offset, bottom_1_data+offset);
    dot += dot_data[idx];
  }
  Dtype loss = 1 - dot / bottom[0]->num();
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void CosLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* dot_data = this->dot_.cpu_data();
    const Dtype* bottom_0_data = bottom[0]->cpu_data();
    Dtype* bottom_0_diff = bottom[0]->mutable_cpu_diff();
    int num = bottom[0]->num();
    int d = bottom[0]->count(1);
    int offset = 0;
    const Dtype beta = -Dtype(1) / num;
    caffe_copy(bottom[0]->count(),
	       bottom[1]->cpu_data(), bottom_0_diff);
    for (int idx = 0; idx < num; ++idx) {
      offset = bottom[0]->offset(idx);
      const Dtype alpha = dot_data[idx] / num;
      caffe_cpu_axpby(
	d,
	alpha,
	bottom_0_data+offset,
	beta,
	bottom_0_diff+offset);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(CosLossLayer);
#endif

INSTANTIATE_CLASS(CosLossLayer);
REGISTER_LAYER_CLASS(CosLoss);

}  // namespace caffe
