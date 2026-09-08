# C++ Neural Network From Scratch

A small neural network library implemented from scratch in modern C++.

The goal of this project is to understand the mechanics behind neural networks by implementing the fundamental components directly, including tensor operations, dense layers, activation functions, forward propagation, backpropagation, loss calculation, and parameter updates.

No machine learning frameworks are used.

## Project Goals

This project is intended to explore both neural network fundamentals and modern C++ design.

The neural network implementation will eventually support:

* 2D tensor and matrix operations
* Dense / fully connected layers
* Activation functions
* Forward propagation
* Backpropagation
* Loss functions
* Gradient descent
* Trainable weights and biases
* Sequential neural network construction
* Simple classification and regression problems

The initial target is to train a small feed-forward neural network capable of learning XOR.

## Neural Network Overview

A basic feed-forward neural network can be represented as:

```text
Input
  |
  v
Linear
  |
  v
Activation
  |
  v
Linear
  |
  v
Prediction
  |
  v
Loss
```

Each linear layer performs the affine transformation:

```text
Y = XW + b
```

where:

* `X` is the input tensor
* `W` contains the trainable weights
* `b` contains the trainable biases
* `Y` is the layer output

Activation functions introduce non-linearity between linear transformations.

Training consists of four main stages:

```text
Forward Pass
    |
    v
Loss Calculation
    |
    v
Backpropagation
    |
    v
Parameter Update
```

During backpropagation, gradients are propagated through the network using the chain rule.

For a simple gradient descent update:

```text
W = W - learning_rate * dL/dW
```

## Architecture

The project is organized around several small classes.

```text
Tensor
  |
  +-- stores numerical data
  +-- matrix operations
  +-- tensor arithmetic

Layer
  |
  +-- common neural network layer interface
       |
       +-- Linear
       +-- ReLU
       +-- future activation/layer types

NeuralNetwork
  |
  +-- stores layers
  +-- performs forward propagation
  +-- performs backward propagation

Loss
  |
  +-- calculates prediction error
  +-- produces the initial loss gradient
```

## Tensor

`Tensor` is currently implemented as a two-dimensional matrix backed by a contiguous:

```cpp
std::vector<double>
```

For example:

```cpp
Tensor tensor{
    {1.0, 2.0},
    {3.0, 4.0}
};
```

is represented internally as:

```text
[1.0, 2.0, 3.0, 4.0]
```

with:

```text
rows = 2
cols = 2
```

An element at row `r` and column `c` is stored at:

```text
r * cols + c
```

### Current Tensor Operations

The current `Tensor` implementation supports:

* Construction from dimensions
* Construction from initializer lists
* Element access
* Shape queries
* Matrix transpose
* Tensor addition
* Tensor subtraction
* Scalar multiplication
* Matrix multiplication
* Row-vector addition
* Row summation
* Stream output for debugging
* Shape and bounds validation

Example:

```cpp
Tensor a{
    {1.0, 2.0},
    {3.0, 4.0}
};

Tensor b{
    {5.0, 6.0},
    {7.0, 8.0}
};

Tensor c = matmul(a, b);

std::cout << c;
```

produces:

```text
[19 22]
[43 50]
```

## Matrix Multiplication

Matrix multiplication is implemented explicitly using:

```cpp
Tensor matmul(const Tensor& lhs,
              const Tensor& rhs);
```

rather than overloading `operator*`.

This keeps matrix multiplication distinct from element-wise multiplication.

For:

```text
A: m x n
B: n x p
```

the result has shape:

```text
C: m x p
```

Each element is calculated as:

```text
C[i][j] = sum(A[i][k] * B[k][j])
```

## Bias Broadcasting

Dense layers require the operation:

```text
Y = XW + b
```

If a batch output has shape:

```text
batch_size x output_features
```

and the bias tensor has shape:

```text
1 x output_features
```

the bias must be added to every row.

The tensor class currently supports this using:

```cpp
tensor.add_row_vector(bias);
```

General-purpose broadcasting is intentionally not implemented yet.

## Row Summation

Backpropagation through a dense layer requires calculating the bias gradient by summing the output gradients across the batch.

For example:

```text
[1 2 3]
[4 5 6]
[7 8 9]
```

becomes:

```text
[12 15 18]
```

using:

```cpp
Tensor result = tensor.sum_rows();
```

This will be used by the dense layer as:

```cpp
grad_bias_ = grad_output.sum_rows();
```

## Planned Layer Interface

All neural network layers will derive from a common abstract interface:

```cpp
class Layer {
public:
    virtual Tensor forward(const Tensor& input) = 0;

    virtual Tensor backward(
        const Tensor& grad_output) = 0;

    virtual void update(double learning_rate) {}

    virtual ~Layer() = default;
};
```

This allows a neural network to store different layer implementations using polymorphism:

```cpp
std::vector<std::unique_ptr<Layer>> layers_;
```

For example:

```text
Linear
ReLU
Linear
ReLU
Linear
```

can all be stored and processed through the same `Layer` interface.

## Planned Linear Layer

A dense layer will contain:

```cpp
Tensor weights_;
Tensor bias_;

Tensor grad_weights_;
Tensor grad_bias_;

Tensor cached_input_;
```

The forward pass performs:

```text
output = input * weights + bias
```

or:

```cpp
Tensor output = matmul(input, weights_);
output.add_row_vector(bias_);
```

During backpropagation:

```text
grad_weights = input^T * grad_output

grad_bias = sum_rows(grad_output)

grad_input = grad_output * weights^T
```

The returned `grad_input` is then propagated into the previous layer.

## Planned ReLU Layer

ReLU applies:

```text
ReLU(x) = max(0, x)
```

During the backward pass:

```text
gradient = grad_output    if input > 0
gradient = 0              otherwise
```

Unlike `Linear`, ReLU contains no trainable parameters.

## Planned Training Loop

Once the core classes are complete, training should look roughly like:

```cpp
NeuralNetwork net;

net.add<Linear>(2, 4);
net.add<ReLU>();
net.add<Linear>(4, 1);

MSELoss loss_fn;

for (int epoch = 0; epoch < epochs; ++epoch) {
    Tensor prediction = net.forward(X);

    double loss =
        loss_fn.forward(prediction, y);

    Tensor grad =
        loss_fn.backward(prediction, y);

    net.backward(grad);

    net.update(learning_rate);
}
```

The implementation therefore closely follows the conceptual training process:

```text
Data
 |
 v
Forward Pass
 |
 v
Prediction
 |
 v
Loss
 |
 v
Backward Pass
 |
 v
Gradients
 |
 v
Update Weights
```

## Project Structure

Planned structure:

```text
.
├── CMakeLists.txt
├── README.md
├── include
│   ├── Layer.hpp
│   ├── Linear.hpp
│   ├── MSELoss.hpp
│   ├── NeuralNetwork.hpp
│   ├── ReLU.hpp
│   └── Tensor.hpp
├── src
│   ├── Linear.cpp
│   ├── MSELoss.cpp
│   ├── NeuralNetwork.cpp
│   ├── ReLU.cpp
│   ├── Tensor.cpp
│   └── main.cpp
└── tests
    └── TensorTests.cpp
```

## Roadmap

Current implementation:

* [x] 2D tensor storage
* [x] Element access
* [x] Shape validation
* [x] Tensor addition
* [x] Tensor subtraction
* [x] Scalar multiplication
* [x] Matrix transpose
* [x] Matrix multiplication
* [x] Row-vector bias addition
* [x] Row summation
* [x] Tensor stream output

Next steps:

* [ ] Tensor unit tests
* [ ] `Layer` interface
* [ ] `Linear` layer
* [ ] Weight initialization
* [ ] `ReLU` activation
* [ ] Mean squared error loss
* [ ] Backpropagation
* [ ] Gradient descent updates
* [ ] `NeuralNetwork` / sequential container
* [ ] XOR training example

Future extensions:

* [ ] Sigmoid
* [ ] Softmax
* [ ] Cross-entropy loss
* [ ] SGD optimizer class
* [ ] Adam optimizer
* [ ] Mini-batch training
* [ ] General N-dimensional tensors
* [ ] Automatic differentiation
* [ ] Additional initialization strategies
* [ ] Performance benchmarking
* [ ] SIMD / parallelization experiments

## Design Philosophy

The implementation intentionally favors clarity over performance.

The first version uses straightforward algorithms so that the relationship between the C++ implementation and the underlying mathematics remains visible.

Optimizations such as cache-aware matrix multiplication, SIMD, multithreading, expression templates, and external BLAS libraries can be explored after the network is functionally correct.

The project is therefore both a neural network implementation and an exercise in understanding how machine learning frameworks represent numerical computation internally.

