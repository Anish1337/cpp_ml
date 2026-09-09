#pragma once

#include "Tensor.hpp"

class Layer {
    public:
        virtual Tensor forward(const Tensor& input) = 0;
        virtual Tensor backward(const Tensor* grad_output) = 0;

        virtual void update(double learning_rate){}

        virtual ~Layer() = default;
};
