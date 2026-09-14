#pragma once
#include "Layer.hpp"
#include <concepts>
#include <memory>
#include <utility>
#include <vector>

class NeuralNetwork {
public:
    template <typename T, typename... Args> requires std::derived_from<T, Layer>
    T& add(Args&&... args)
    {
        auto layer = std::make_unique<T>(std::forward<Args>(args)...);
        T& reference = *layer;
        layers_.push_back(std::move(layer));
        return reference;
    }
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);
    void update(double learning_rate);

private:
    std::vector<std::unique_ptr<Layer>> layers_;
};
