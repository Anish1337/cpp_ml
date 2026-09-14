#include "ReLU.hpp"
#include <stdexcept>

Tensor ReLU::forward(const Tensor& input)
{
    cached_input_ = input;

    Tensor output(input.rows(), input.cols());

    for (std::size_t i = 0; i < input.rows(); ++i) {
        for (std::size_t j = 0; j < input.cols(); ++j) {
            output(i, j) = input(i, j) > 0.0
                ? input(i, j)
                : 0.0;
        }
    }

    return output;
}

Tensor ReLU::backward(const Tensor& grad_output)
{
    if (!cached_input_) {
        throw std::logic_error("ReLU backward requires forward first");
    }
    if (grad_output.rows() != cached_input_->rows() ||
        grad_output.cols() != cached_input_->cols()) {
        throw std::invalid_argument("ReLU gradient shape mismatch");
    }
    Tensor grad_input(grad_output.rows(), grad_output.cols());

    for (std::size_t i = 0; i < grad_output.rows(); ++i) {
        for (std::size_t j = 0; j < grad_output.cols(); ++j) {
            grad_input(i, j) = (*cached_input_)(i, j) > 0.0
                ? grad_output(i, j)
                : 0.0;
        }
    }

    return grad_input;
}
