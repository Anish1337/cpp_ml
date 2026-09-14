#include "Linear.hpp"
#include "ReLU.hpp"
#include "MSELoss.hpp"
#include "NeuralNetwork.hpp"
#include <cmath>
#include <iostream>

int main()
{
    const Tensor inputs{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const Tensor targets{{0}, {1}, {1}, {0}};
    NeuralNetwork network;
    network.add<Linear>(2, 8, 42);
    network.add<ReLU>();
    network.add<Linear>(8, 1, 43);
    MSELoss loss;
    const double initial = loss.forward(network.forward(inputs), targets);
    for (int epoch = 0; epoch < 10000; ++epoch) {
        const Tensor prediction = network.forward(inputs);
        network.backward(loss.backward(prediction, targets));
        network.update(0.05);
    }
    const Tensor prediction = network.forward(inputs);
    const double final = loss.forward(prediction, targets);
    std::cout << "Initial MSE: " << initial << "\nFinal MSE: " << final
              << "\nPredictions (00, 01, 10, 11):\n" << prediction;
    if (!std::isfinite(final) || final >= 0.001 || final >= initial) return 1;
    for (std::size_t r = 0; r < targets.rows(); ++r)
        if (!std::isfinite(prediction(r, 0)) ||
            std::abs(prediction(r, 0) - targets(r, 0)) >= 0.1) return 1;
    std::cout << "XOR passed\n";
}
