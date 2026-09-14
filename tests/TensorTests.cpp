#include "Tensor.hpp"
#include "Linear.hpp"
#include "ReLU.hpp"
#include "MSELoss.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

void near(double actual, double expected, double tolerance = 1e-6)
{
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance)
        throw std::runtime_error("Numerical comparison failed");
}

template <typename Exception, typename Function>
void throws(Function function)
{
    try { function(); }
    catch (const Exception&) { return; }
    throw std::runtime_error("Expected exception was not thrown");
}

int main()
{
    try {
        Tensor a{{1, 2}, {3, 4}};
        Tensor b{{5, 6}, {7, 8}};
        const Tensor product = matmul(a, b);
        near(product(0, 0), 19); near(product(0, 1), 22);
        near(product(1, 0), 43); near(product(1, 1), 50);
        near(a.transpose()(0, 1), 3);
        near(a.sum_rows()(0, 1), 6);
        near((a + b)(1, 1), 12);
        near((b - a)(0, 0), 4);
        near((2.0 * a)(1, 0), 6);
        a.add_row_vector(Tensor{{10, 20}});
        near(a(0, 0), 11); near(a(1, 1), 24);
        throws<std::out_of_range>([&] { (void)a(2, 0); });
        throws<std::invalid_argument>([&] { (void)matmul(a, Tensor(3, 1)); });
        throws<std::invalid_argument>([&] { a.add_row_vector(Tensor(2, 2)); });
        throws<std::invalid_argument>([] { Tensor bad{{1}, {2, 3}}; });

        MSELoss loss;
        Tensor x{{-0.7, 0.2}, {0.4, 1.3}};
        const Tensor y{{0.3, -0.2}, {0.1, 0.8}};
        constexpr double epsilon = 1e-6;
        const Tensor loss_gradient = loss.backward(x, y);
        for (std::size_t r = 0; r < x.rows(); ++r)
            for (std::size_t c = 0; c < x.cols(); ++c) {
                Tensor plus = x, minus = x;
                plus(r, c) += epsilon; minus(r, c) -= epsilon;
                near(loss_gradient(r, c),
                     (loss.forward(plus, y) - loss.forward(minus, y)) / (2 * epsilon));
            }
        throws<std::invalid_argument>([&] { loss.forward(Tensor(0, 2), Tensor(0, 2)); });
        throws<std::invalid_argument>([&] { loss.backward(x, Tensor(1, 2)); });

        ReLU relu;
        throws<std::logic_error>([&] { relu.backward(x); });
        const Tensor activated = relu.forward(x);
        near(activated(0, 0), 0); near(activated(1, 1), 1.3);
        const Tensor relu_gradient = relu.backward(Tensor{{1, 1}, {1, 1}});
        for (std::size_t r = 0; r < x.rows(); ++r)
            for (std::size_t c = 0; c < x.cols(); ++c) {
                Tensor plus = x, minus = x;
                plus(r, c) += epsilon; minus(r, c) -= epsilon;
                const double high = relu.forward(plus)(r, c);
                const double low = relu.forward(minus)(r, c);
                near(relu_gradient(r, c), (high - low) / (2 * epsilon));
            }
        throws<std::invalid_argument>([&] { relu.backward(Tensor(1, 1)); });

        Linear linear(2, 2, 7);
        throws<std::logic_error>([&] { linear.backward(y); });
        throws<std::logic_error>([&] { linear.update(0.1); });
        const Tensor prediction = linear.forward(x);
        const Tensor dx = linear.backward(loss.backward(prediction, y));
        const Tensor dw = linear.grad_weights(), db = linear.grad_bias();
        // Perturb inputs and parameters independently to check the chain rule.
        for (std::size_t r = 0; r < 2; ++r)
            for (std::size_t c = 0; c < 2; ++c) {
                Tensor plus = x, minus = x;
                plus(r, c) += epsilon; minus(r, c) -= epsilon;
                near(dx(r, c), (loss.forward(linear.forward(plus), y) -
                     loss.forward(linear.forward(minus), y)) / (2 * epsilon));
                auto objective = [&](double delta) {
                    Tensor weights = linear.weights();
                    weights(r, c) += delta;
                    Tensor output = matmul(x, weights);
                    output.add_row_vector(linear.bias());
                    return loss.forward(output, y);
                };
                near(dw(r, c), (objective(epsilon) - objective(-epsilon)) / (2 * epsilon));
            }
        for (std::size_t c = 0; c < 2; ++c) {
            auto objective = [&](double delta) {
                Tensor bias = linear.bias();
                bias(0, c) += delta;
                Tensor output = matmul(x, linear.weights());
                output.add_row_vector(bias);
                return loss.forward(output, y);
            };
            near(db(0, c), (objective(epsilon) - objective(-epsilon)) / (2 * epsilon));
        }
        const double before = loss.forward(linear.forward(x), y);
        linear.backward(loss.backward(linear.forward(x), y));
        linear.update(0.01);
        if (loss.forward(linear.forward(x), y) >= before)
            throw std::runtime_error("Gradient update did not lower loss");
        throws<std::invalid_argument>([&] { linear.backward(Tensor(1, 1)); });
        std::cout << "Tensor, layer, loss, and gradient checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
