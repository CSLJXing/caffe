#ifndef CAFFE_SOFTMAX_CROSS_ENTROPY_LOSS_LAYER_HPP_
#define CAFFE_SOFTMAX_CROSS_ENTROPY_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"

namespace caffe {

/**
 * @brief Computes the multinomial logistic loss between two
 *        vectors , passing real-valued predictions through a
 *        softmax to get a probability distribution over classes.
 *
 * This layer should be preferred over separate
 * SoftmaxLayer + CrossEntropyLayer
 * as its gradient computation is more numerically stable.
 * At test time, this layer can be replaced simply by a SoftmaxLayer.
 *
 * @param bottom input Blob vector (length 2)
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the predictions @f$ x @f$, a Blob with values in
 *      @f$ [-\infty, +\infty] @f$ indicating the predicted score for each of
 *      the @f$ K = CHW @f$ classes. This layer maps these scores to a
 *      probability distribution over classes using the softmax function
 *      @f$ \hat{p}_{nk} = \exp(x_{nk}) /
 *      \left[\sum_{k'} \exp(x_{nk'})\right] @f$ (see SoftmaxLayer).
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the targets @f$ y @f$, a Blob with values in
 *      @f$ [-\infty, +\infty] @f$ indicating the target score for each of
 *      the @f$ K = CHW @f$ classes. This layer maps these scores to a
 *      probability distribution over classes using the softmax function
 *      @f$ q_{nk} = \exp(y_{nk}) /
 *      \left[\sum_{k'} \exp(y_{nk'})\right] @f$ (see SoftmaxLayer).
 * @param top output Blob vector (length 1)
 *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
 *      the computed cross-entropy classification loss: @f$ E =
 *        \frac{-1}{N} \sum\limits_{n=1}^N \sum\limits_{j=1}^K q_{nk} \log(\hat{p}_{nk})
 *      @f$, for softmax output class probabilites @f$ \hat{p} @f$
 */
template <typename Dtype>
class SoftmaxWithCrossEntropyLossLayer : public LossLayer<Dtype> {
 public:
  /**
   * @param param provides LossParameter loss_param, with options:
   *  - normalize (optional, default true)
   *    If true, the loss is normalized by the number of (nonignored) labels
   *    present; otherwise the loss is simply summed over spatial locations.
   */
  explicit SoftmaxWithCrossEntropyLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SoftmaxWithCrossEntropyLoss"; }

 protected:
  /// @copydoc SoftmaxWithCrossEntropyLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  /**
   * @brief Computes the softmax loss error gradient w.r.t. the predictions.
   *
   * Gradients cannot be computed with respect to the target inputs (bottom[1]),
   * so this method ignores bottom[1] and requires !propagate_down[1], crashing
   * if propagate_down[1] is set.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *      respect to the outputs
   *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
   *      This Blob's diff will simply contain the loss_weight* @f$ \lambda @f$,
   *      as @f$ \lambda @f$ is the coefficient of this layer's output
   *      @f$\ell_i@f$ in the overall Net loss
   *      @f$ E = \lambda_i \ell_i + \mbox{other loss terms}@f$; hence
   *      @f$ \frac{\partial E}{\partial \ell_i} = \lambda_i @f$.
   *      (*Assuming that this top Blob is not used as a bottom (input) by any
   *      other layer of the Net.)
   * @param propagate_down see Layer::Backward.
   *      propagate_down[1] must be false as we can't compute gradients with
   *      respect to the targets.
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the predictions @f$ x @f$; Backward computes diff
   *      @f$ \frac{\partial E}{\partial x} @f$
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the targets -- ignored as we can't compute their error gradients
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


  /// The internal SoftmaxLayer used to map predictions to a distribution.
  shared_ptr<Layer<Dtype> > softmax_layer_;
  /// prob stores the output probability predictions from the SoftmaxLayer.
  Blob<Dtype> predict_prob_;
  Blob<Dtype> target_prob_;
  /// bottom vector holder used in call to the underlying SoftmaxLayer::Forward
  vector<Blob<Dtype>*> softmax_predict_bottom_vec_;
  vector<Blob<Dtype>*> softmax_target_bottom_vec_;
  /// top vector holder used in call to the underlying SoftmaxLayer::Forward
  vector<Blob<Dtype>*> softmax_predict_top_vec_;
  vector<Blob<Dtype>*> softmax_target_top_vec_;
  /// Whether to normalize the loss by the total number of values present
  /// (otherwise just by the batch size).
  bool normalize_;

  int softmax_axis_, outer_num_, inner_num_;
};

}  // namespace caffe

#endif //  CAFFE_SOFTMAX_CROSS_ENTROPY_LOSS_LAYER_HPP_
