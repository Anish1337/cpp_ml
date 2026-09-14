#include "Linear.hpp"
#include <cmath>
#include <stdexcept>

Linear::Linear(std::size_t inputs, std::size_t outputs, std::uint32_t seed)
    : weights_(inputs, outputs), bias_(1, outputs),
      grad_weights_(inputs, outputs), grad_bias_(1, outputs)
{
    if (inputs == 0 || outputs == 0) {
        throw std::invalid_argument("Linear dimensions must be positive");
    }
    std::mt19937 generator(seed);
    // He uniform initialization for ReLU networks. Biases start at zero.
    const double bound = std::sqrt(6.0 / static_cast<double>(inputs));
    std::uniform_real_distribution<double> distribution(-bound, bound);
    for (std::size_t r = 0; r < inputs; ++r)
        for (std::size_t c = 0; c < outputs; ++c)
            weights_(r, c) = distribution(generator);
}

Tensor Linear::forward(const Tensor& input)
{
    Tensor output = matmul(input, weights_);
    output.add_row_vector(bias_);
    cached_input_ = input;
    gradients_ready_ = false;
    return output;
}

Tensor Linear::backward(const Tensor& grad_output)
{
    if (!cached_input_) {
        throw std::logic_error("Linear backward requires forward first");
    }
    if (grad_output.rows() != cached_input_->rows() ||
        grad_output.cols() != weights_.cols()) {
        throw std::invalid_argument("Linear gradient shape mismatch");
    }
    grad_weights_ = matmul(cached_input_->transpose(), grad_output);
    grad_bias_ = grad_output.sum_rows();
    gradients_ready_ = true;
    return matmul(grad_output, weights_.transpose());
}

void Linear::update(double learning_rate)
{
    if (!std::isfinite(learning_rate) || learning_rate < 0.0) {
        throw std::invalid_argument("Learning rate must be finite and nonnegative");
    }
    if (!gradients_ready_) {
        throw std::logic_error("Linear update requires backward first");
    }
    weights_ -= learning_rate * grad_weights_;
    bias_ -= learning_rate * grad_bias_;
    gradients_ready_ = false;
    cached_input_.reset();
}
