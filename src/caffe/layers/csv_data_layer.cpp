#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/csv_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
CSVDataLayer<Dtype>::~CSVDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void CSVDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Read the file with filenames and labels
  const string& source = this->layer_param_.csv_data_param().source();
  const int num_feature = this->layer_param_.csv_data_param().num_feature();
  CHECK_GT(num_feature, 0) << "The number of feature column must be larger than 0.";
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string line;
  size_t pos_1, pos_2;
  while (std::getline(infile, line)) {
    std::vector<float> cols;
    pos_2 = -1;
    pos_1 = line.find_first_of(' ');
    while (pos_1 != std::string::npos) {
      cols.push_back(std::atof(line.substr(pos_2+1, pos_1-pos_2-1).c_str()));
      pos_2 = pos_1;
      pos_1 = line.find_first_of(' ', pos_1+1);
    }
    cols.push_back(std::atof(line.substr(pos_2+1).c_str()));
    CHECK_GT(cols.size(), num_feature) << "The number of columns must be larger than num_feature.";
    lines_.push_back(cols);
  }

  CHECK(!lines_.empty()) << "File is empty";

  if (this->layer_param_.csv_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleLines();
  } else {
    if (this->phase_ == TRAIN && Caffe::solver_rank() > 0 &&
        this->layer_param_.csv_data_param().rand_skip() == 0) {
      LOG(WARNING) << "Shuffling or skipping recommended for multi-GPU";
    }
  }
  LOG(INFO) << "A total of " << lines_.size() << " records.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.csv_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.csv_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }

  const int batch_size = this->layer_param_.csv_data_param().batch_size();
  const int num_target = lines_[0].size() - num_feature;
  vector<int> feature_shape, target_shape;
  feature_shape.push_back(batch_size);
  feature_shape.push_back(num_feature);
  top[0]->Reshape(feature_shape);
  target_shape.push_back(batch_size);
  target_shape.push_back(num_target);
  top[1]->Reshape(target_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(feature_shape);
    this->prefetch_[i]->label_.Reshape(target_shape);
  }
}

template <typename Dtype>
void CSVDataLayer<Dtype>::ShuffleLines() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void CSVDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  CHECK(batch->data_.count());
  CHECK(batch->label_.count());
  CSVDataParameter csv_data_param = this->layer_param_.csv_data_param();
  const int batch_size = csv_data_param.batch_size();
  const int num_feature = csv_data_param.num_feature();
  const int num_target = lines_[0].size() - num_feature;

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    CHECK_GT(lines_size, lines_id_);
    int offset = batch->data_.offset(item_id);
    Dtype* this_data = prefetch_data + offset;
    for (int idx = 0; idx < num_feature; ++idx) {
      this_data[idx] = lines_[lines_id_][idx];
    }
    offset = batch->label_.offset(item_id);
    Dtype* this_label = prefetch_label + offset;
    for (int idx = 0; idx < num_target; ++idx) {
      this_label[idx] = lines_[lines_id_][idx+num_feature];
    }
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.csv_data_param().shuffle()) {
        ShuffleLines();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
}

INSTANTIATE_CLASS(CSVDataLayer);
REGISTER_LAYER_CLASS(CSVData);

}  // namespace caffe
