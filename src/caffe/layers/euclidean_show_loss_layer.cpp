#include <vector>

#include "caffe/layers/euclidean_show_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanShowLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  vector<int> top_shape;
  top_shape.push_back(diff_.shape(1));
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void EuclideanShowLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  caffe_powx(
     count,
     diff_.cpu_data(),
     Dtype(2),
     diff_.mutable_cpu_data());
  const int num_channel = diff_.shape(1), batch_size = diff_.shape(0);
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int idx = 0; idx < num_channel; ++idx) {
    top_data[idx] = 0;
    for (int jdx = 0; jdx < batch_size; ++jdx) {
      top_data[idx] += diff_.data_at(jdx, idx, 0, 0);
    }
    top_data[idx] /= batch_size;
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanShowLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanShowLossLayer);
REGISTER_LAYER_CLASS(EuclideanShowLoss);

}  // namespace caffe
