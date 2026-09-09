#pragma once

#include "Layer.hpp"

class ReLU : public Layer {
public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;

private:
    Tensor input_cache;
};
