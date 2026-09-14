#pragma once
#include "Layer.hpp"
#include <cstdint>
#include <optional>
#include <random>

class Linear : public Layer {
public:
    Linear(std::size_t inputs, std::size_t outputs,
           std::uint32_t seed = std::random_device{}());
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    void update(double learning_rate) override;

    const Tensor& weights() const noexcept { return weights_; }
    const Tensor& bias() const noexcept { return bias_; }
    const Tensor& grad_weights() const noexcept { return grad_weights_; }
    const Tensor& grad_bias() const noexcept { return grad_bias_; }

private:
    Tensor weights_, bias_, grad_weights_, grad_bias_;
    std::optional<Tensor> cached_input_;
    bool gradients_ready_ = false;
};
