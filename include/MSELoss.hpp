#pragma once

#include "Tensor.hpp"

class MSELoss {
public:
    // compute MSE
    [[nodiscard]]
    double forward(const Tensor& prediction,
                   const Tensor& target) const;
    // compute grad for backpropagation
    [[nodiscard]]
    Tensor backward(const Tensor& prediction,
                    const Tensor& target) const;

private:
    static void check_same_shape(
        const Tensor& prediction,
        const Tensor& target);
};
