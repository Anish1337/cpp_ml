#pragma once

#include "Layer.hpp"
#include <optional>

class ReLU : public Layer {
public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;

private:
    std::optional<Tensor> cached_input_;
};
