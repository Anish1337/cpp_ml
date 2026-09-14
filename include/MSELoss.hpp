#pragma once
#include "Tensor.hpp"

class MSELoss {
public:
    double forward(const Tensor& prediction, const Tensor& target) const;
    Tensor backward(const Tensor& prediction, const Tensor& target) const;
};
