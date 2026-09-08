#include "MSELoss.hpp"

#include <stdexcept>

void MSELoss::check_same_shape(
    const Tensor& prediction,
    const Tensor& target)
{
    if (prediction.rows() != target.rows() ||
        prediction.cols() != target.cols()) {
        throw std::invalid_argument{
            "MSELoss: prediction and target shapes must match"
        };
    }
}

// L = (1 / N) * sum((prediction - target)^2)
double MSELoss::forward(
    const Tensor& prediction,
    const Tensor& target) const
{
    check_same_shape(prediction, target);

    if (prediction.size() == 0) {
        throw std::invalid_argument{
            "MSELoss: tensors cannot be empty"
        };
    }

    double sum = 0.0;

    for (std::size_t r = 0; r < prediction.rows(); ++r) {
        for (std::size_t c = 0; c < prediction.cols(); ++c) {

            const double diff =
                prediction(r, c) - target(r, c);

            sum += diff * diff;
        }
    }

    return sum /
           static_cast<double>(prediction.size());
}

// dL/dpred = (2 / N) * (pred - target)
Tensor MSELoss::backward(
    const Tensor& prediction,
    const Tensor& target) const
{
    check_same_shape(prediction, target);

    if (prediction.size() == 0) {
        throw std::invalid_argument{
            "MSELoss: tensors cannot be empty"
        };
    }

    const double scale =
        2.0 /
        static_cast<double>(prediction.size());

    return scale * (prediction - target);
}
