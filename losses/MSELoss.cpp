#include "MSELoss.hpp"
#include <stdexcept>

namespace {
void validate(const Tensor& prediction, const Tensor& target)
{
    if (prediction.rows() != target.rows() || prediction.cols() != target.cols()
        || prediction.size() == 0) {
        throw std::invalid_argument("MSE requires matching, nonempty tensors");
    }
}
}

double MSELoss::forward(const Tensor& prediction, const Tensor& target) const
{
    validate(prediction, target);
    double total = 0.0;
    for (std::size_t r = 0; r < prediction.rows(); ++r)
        for (std::size_t c = 0; c < prediction.cols(); ++c) {
            const double error = prediction(r, c) - target(r, c);
            total += error * error;
        }
    return total / static_cast<double>(prediction.size());
}

Tensor MSELoss::backward(const Tensor& prediction, const Tensor& target) const
{
    validate(prediction, target);
    return (prediction - target) * (2.0 / static_cast<double>(prediction.size()));
}
