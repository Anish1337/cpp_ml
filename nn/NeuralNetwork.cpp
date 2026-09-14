#include "NeuralNetwork.hpp"

Tensor NeuralNetwork::forward(const Tensor& input)
{
    Tensor output = input;
    for (auto& layer : layers_) output = layer->forward(output);
    return output;
}

Tensor NeuralNetwork::backward(const Tensor& grad_output)
{
    Tensor gradient = grad_output;
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it)
        gradient = (*it)->backward(gradient);
    return gradient;
}

void NeuralNetwork::update(double learning_rate)
{
    for (auto& layer : layers_) layer->update(learning_rate);
}
