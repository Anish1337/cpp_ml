#pragma once

#include <"Layer.hpp">

class Linear : public Layer{
    public:
        Linear(std::size_t in_feat, std::size_t out_feat);

        Tensor forward(const Tensor& input) override;
        Tensor backward(const Tensor& grad_output) override;

        void update(double learning_rate) override;
    private:
        Tensor weights_;
        Tensor bias_;

        Tensor grade_weights_;
        Tensor grad_bias_;

        Tensor cached_input_;
}
