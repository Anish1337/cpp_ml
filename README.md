# C++ Machine Learning From Scratch

A small C++23 machine learning library implemented using only the C++ standard
library. Tensor arithmetic, layers, derivatives, and parameter updates are written
explicitly so the connection between the mathematics and code stays visible.

The first complete example trains a feed-forward neural network to learn XOR.
The numerical core is independent of neural networks, allowing other ML
algorithms to reuse it later. The project currently favors clarity over speed.

## Build and run

Run the following commands from the `cpp_ml` project directory. The code uses
C++23 as its configured language standard and `std::cout` for output.

### With CMake

Requires a C++23 compiler and CMake 3.20 or newer. No external libraries are
required. Use a fresh build directory if an existing CMake cache refers to an
old checkout location.

```bash
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug

# Run XOR training.
./build-debug/cpp_ml_example

# Run component checks directly, or run both tests through CTest.
./build-debug/cpp_ml_tests
ctest --test-dir build-debug --output-on-failure
```

CMake builds a `cpp_ml` library and links the example and test executables to it.
Tests are enabled by default; `-DBUILD_TESTING=OFF` disables the component test
target and CTest registration.

### Directly with GCC

Use `g++`, GCC's C++ driver, to compile and link the C++ standard library.

```bash
g++ -std=c++23 -O2 -Wall -Wextra -Wpedantic \
    -Icore -Iinclude \
    core/Tensor.cpp nn/Linear.cpp nn/ReLU.cpp nn/NeuralNetwork.cpp \
    losses/MSELoss.cpp examples/xor.cpp \
    -o xor

./xor
```

Build the component tests separately:

```bash
g++ -std=c++23 -O2 -Wall -Wextra -Wpedantic \
    -Icore -Iinclude \
    core/Tensor.cpp nn/Linear.cpp nn/ReLU.cpp nn/NeuralNetwork.cpp \
    losses/MSELoss.cpp tests/TensorTests.cpp \
    -o nn_tests

./nn_tests
```

`-Icore -Iinclude` supplies the header search directories; headers do not need to
be compiled separately. The example and tests each define `main()`, so they
belong in separate executables. `./` runs an executable in the current directory.

## File structure

```text
cpp_ml/
├── CMakeLists.txt          Library, executables, language standard, and CTest setup
├── README.md
├── core/
│   ├── Tensor.hpp         2D numerical container and operation declarations
│   └── Tensor.cpp         Storage, validation, arithmetic, and matrix operations
├── include/
│   ├── Layer.hpp          Abstract layer interface
│   ├── Linear.hpp         Dense layer interface and owned training state
│   ├── ReLU.hpp           Activation interface and input cache
│   ├── MSELoss.hpp        Mean squared error interface
│   └── NeuralNetwork.hpp  Sequential container and templated layer construction
├── nn/
│   ├── Linear.cpp         Initialization, affine transform, gradients, and updates
│   ├── ReLU.cpp           Activation and derivative
│   └── NeuralNetwork.cpp  Forward, reverse, and update traversals
├── losses/
│   └── MSELoss.cpp        Scalar loss and prediction gradient
├── examples/
│   └── xor.cpp            Data, network setup, training, and success criteria
└── tests/
    └── TensorTests.cpp    Tensor, layer, loss, and numerical gradient checks
```

`build-debug/` is generated build output. Tensor's header currently lives beside
its implementation, while neural network and loss headers live in `include/`.
CMake exports both header directories to targets that link the library.

`core/` provides numerical operations without depending on ML algorithms. `nn/`
implements neural network behavior using that core. `losses/` measures prediction
error and can also serve future models. `examples/` owns application-specific
data and training choices; those do not belong in the reusable library.

## Architecture

A network is an ordered sequence of layers. Each layer transforms a batch of
values and knows how to propagate a loss gradient through its own operation.

```text
Forward:
inputs → Linear(2, 8) → ReLU → Linear(8, 1) → predictions
                                                  │
                                      targets → MSELoss
                                                  │
Backward:                              prediction gradient
                                                  │
          Linear(2, 8) ← ReLU ← Linear(8, 1) ←───────┘

Update: each Linear layer applies its stored parameter gradients.
```

Rows represent examples; columns represent features. For XOR, all four examples
are processed as one batch:

| Stage | Shape | Meaning |
| --- | --- | --- |
| Input | `4 × 2` | Four pairs of input bits |
| First Linear output | `4 × 8` | Eight learned values per example |
| ReLU output | `4 × 8` | Nonlinear hidden activations |
| Final Linear output | `4 × 1` | One prediction per example |
| Target | `4 × 1` | Expected XOR outputs |
| MSE | Scalar | Average squared prediction error |

The complete forward expression is `prediction = ReLU(XW₁ + b₁)W₂ + b₂`.
There are 33 trainable parameters: 16 weights and 8 biases in the first layer,
plus 8 weights and 1 bias in the second.

## Classes and functions

### Tensor

[Tensor.hpp](core/Tensor.hpp) declares a 2D matrix backed by
`std::vector<double>`. Values are contiguous in row-major order:

```text
[[1, 2], [3, 4]] → [1, 2, 3, 4]
index(row, col)  = row * cols + col
```

One vector gives the tensor automatic memory ownership and a simple layout,
without a separate allocation for every row. Tensor has value semantics:
copying a tensor copies its data, and returned tensors own their results.

| Function | Responsibility |
| --- | --- |
| `Tensor(rows, cols)` | Allocate a zero-filled matrix |
| Initializer-list constructor | Construct from rows, rejecting unequal row lengths |
| `operator()(row, col)` | Return mutable or const element access with bounds checking |
| `rows()`, `cols()`, `size()` | Report shape and total element count |
| `transpose()` | Return a matrix with rows and columns exchanged |
| `sum_rows()` | Return a `1 × cols` sum across the batch |
| `add_row_vector(row)` | Add one `1 × cols` vector to every row in place |
| `+=`, `-=`, `*=` | Perform in-place addition, subtraction, or scalar multiplication |
| `+`, `-`, scalar `*` | Return arithmetic results as new tensors |
| `matmul(lhs, rhs)` | Multiply `m × n` and `n × p` matrices into an `m × p` result |
| `operator<<` | Print values through an output stream |
| Private `index()` | Validate coordinates and calculate the flat offset |
| Private `check_same_shape()` | Validate element-wise addition and subtraction |

Matrix multiplication has the explicit name `matmul`; `*` currently means scalar
multiplication, not matrix or element-wise tensor multiplication. Bias
broadcasting is deliberately limited to adding a row vector. General broadcasting
and N-dimensional tensors are not implemented.

### Layer

[Layer.hpp](include/Layer.hpp) defines the shared behavior:

```cpp
class Layer {
public:
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;
    virtual void update(double) {}
    virtual ~Layer() = default;
};
```

`forward()` and `backward()` are pure virtual: every concrete layer must provide
both. `backward()` receives the derivative of loss with respect to the layer's
output, then returns the derivative with respect to its input.

`update()` defaults to doing nothing because activations have no trainable
parameters. The virtual destructor ensures derived objects are destroyed
correctly when owned through a `Layer` pointer.

### Linear

[Linear.cpp](nn/Linear.cpp) implements a fully connected layer. Its name follows
common neural network terminology; with a bias, its transformation is affine.

For batch size `B`, input features `I`, and output features `O`:

```text
X: B × I       W: I × O       b: 1 × O
Y = XW + b: B × O
```

| Function | Responsibility |
| --- | --- |
| `Linear(inputs, outputs, seed)` | Allocate parameters and gradients; reject zero feature counts |
| `forward(input)` | Compute `XW + b`, cache the input, and invalidate old gradients |
| `backward(grad_output)` | Calculate parameter gradients and return the input gradient |
| `update(learning_rate)` | Apply gradient descent and consume the cached training state |
| `weights()`, `bias()` | Expose parameters by const reference for inspection |
| `grad_weights()`, `grad_bias()` | Expose stored gradients by const reference for inspection and tests |

The constructor uses `std::mt19937` with uniform weights in
`[-sqrt(6 / inputs), +sqrt(6 / inputs)]`, a He uniform initialization suited to
ReLU hidden layers. This implementation uses it for both Linear layers. Biases
start at zero. Random weights break symmetry so hidden neurons can learn
different features. Without an explicit seed, the constructor obtains one from
`std::random_device`.

Let `G = dL/dY`, the incoming loss gradient. The chain rule gives:

```text
grad_weights = XᵀG       // I × O
grad_bias    = sum_rows(G) // 1 × O
grad_input   = GWᵀ       // B × I
```

`backward()` stores the first two and returns the third. Gradients describe how
a small change to each value would affect loss. Gradient descent subtracts them:

```text
W -= learning_rate * grad_weights
b -= learning_rate * grad_bias
```

The cached input is `std::optional<Tensor>`: absence means no usable forward input
exists. This represents the state explicitly without adding a default constructor
to Tensor. `gradients_ready_` prevents updates before backward or repeated updates
using the same gradients. Update checks for a finite, nonnegative learning rate,
then clears the cache and marks gradients unavailable. Gradient getters do not
perform readiness checks; inspect them immediately after a successful backward.

### ReLU

[ReLU.cpp](nn/ReLU.cpp) introduces nonlinearity without trainable parameters.

```text
forward:  output = max(0, input)
backward: grad_input = grad_output if cached input > 0, otherwise 0
```

`forward()` caches its input and applies the activation element by element.
`backward()` checks that a cache exists and the gradient shape matches. At exactly
zero, the implementation chooses a derivative of zero. ReLU inherits Layer's
no-op `update()`.

Without a nonlinear activation, composing the two affine layers would still
produce one affine transformation, which cannot separate XOR's classes.

### MSELoss

[MSELoss.cpp](losses/MSELoss.cpp) measures prediction error. For `N` total output
elements, rather than just the number of rows:

```text
forward(prediction, target):
    loss = sum((prediction - target)²) / N

backward(prediction, target):
    gradient = 2 * (prediction - target) / N
```

The private-to-file `validate()` helper rejects mismatched shapes and empty
tensors. MSELoss has no parameters or cached state; both public methods are const.
Normalization happens here, so Linear does not divide gradients by batch size
again.

Loss is separate from Layer because it takes both predictions and targets and
returns a scalar for reporting. Keeping it outside the sequence lets the same
network be paired with another objective later.

### NeuralNetwork

[NeuralNetwork.hpp](include/NeuralNetwork.hpp) and
[NeuralNetwork.cpp](nn/NeuralNetwork.cpp) coordinate the layers.

| Function | Responsibility |
| --- | --- |
| `add<T>(args...)` | Construct and append a Layer-derived object, returning a reference to it |
| `forward(input)` | Visit layers in insertion order, passing each result to the next |
| `backward(grad_output)` | Visit layers in reverse order and return the gradient at the network input |
| `update(learning_rate)` | Ask each layer to update its parameters |

An empty network acts as an identity for forward and backward. The current
container supports a sequential chain, not branches or arbitrary computation
graphs. Layer dimensions are checked during numerical operations rather than
when layers are appended.

## Why this OOP and ownership design?

### A network owns a sequence of different layer types

The container is:

```cpp
std::vector<std::unique_ptr<Layer>> layers_;
```

The vector represents the **network's ordered collection of layers**, not one
layer. Each pointer owns one concrete object such as `Linear` or `ReLU`.

- **`std::vector` preserves execution order.** Forward traverses from beginning
  to end; backward traverses the same sequence in reverse. Its size can grow as
  `add()` appends layers.
- **A base-class pointer supports different concrete types.** A `Layer*` can
  refer to either Linear or ReLU. Virtual calls dispatch to the correct
  implementation without type switches in NeuralNetwork.
- **`std::unique_ptr` expresses sole ownership.** The network controls layer
  lifetime. Destruction automatically releases each layer and its tensors, with
  no manual `delete`. Shared ownership and reference counting are unnecessary.
- **Pointers preserve the complete derived objects.** `std::vector<Layer>` is
  impossible because Layer is abstract. Even with a concrete base, storing
  derived objects as base values would slice off their derived state.

The vector stores pointers contiguously; the layer objects themselves are
separately allocated. Growing the vector moves the smart pointers without moving
the layer objects. A reference returned by `add()` therefore survives later
appends, but does not own the layer and must not outlive it. The current network
is not copyable because its unique pointers cannot be copied; independent copies
would require an explicit cloning design.

### Construction uses templates; execution uses virtual functions

```cpp
network.add<Linear>(2, 8, 42);
network.add<ReLU>();
```

`add<T>()` uses `std::derived_from<T, Layer>` to constrain the type at compile
time. `std::forward` passes constructor arguments through to
`std::make_unique<T>`. `std::move` transfers the new pointer's ownership into the
vector. This template lives in the header so callers can instantiate it for each
concrete layer type.

Once constructed, the network executes every layer through the same virtual
interface. Adding a new layer implementation does not require editing the
network traversal code.

### Classes encapsulate the state needed by their operations

Linear owns its parameters, gradients, and cached input. ReLU owns its activation
cache. NeuralNetwork owns the sequence but does not need to understand the
internal math of either layer. Tensor owns numerical data without inheriting
from Layer, and MSELoss remains a separate objective.

This keeps each mathematical operation close to its derivative and state. The
tradeoff is virtual dispatch, separate layer allocations, and tensor copies.
These are acceptable for this educational implementation; it is not designed as
a highly optimized numerical engine. Static composition or other representations
could be explored if measurement later shows a need.

### Forward, backward, and update remain separate

Training follows this contract:

```text
forward → loss gradient → backward through every layer → update every layer
```

Backward must use the weights from the corresponding forward pass. Keeping
updates separate ensures all gradients are computed before any weights change.
For example, `grad_input = G Wᵀ` must use the pre-update `W`.

Each forward replaces the layer's previous cache, and each backward replaces its
stored parameter gradients rather than accumulating them. Inputs are cached as
owned copies, so later changes to the caller's input do not alter the saved
values. There is only one active cache per layer: no concurrent passes, separate
inference mode, or automatic differentiation graph. The container does not make
operations transactional; invalid call sequences can throw after earlier layers
have already been visited.

## How training produces XOR

The example supplies all four input pairs and their desired outputs:

```cpp
const Tensor inputs{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
const Tensor targets{{0}, {1}, {1}, {0}};

NeuralNetwork network;
network.add<Linear>(2, 8, 42);
network.add<ReLU>();
network.add<Linear>(8, 1, 43);
MSELoss loss;

for (int epoch = 0; epoch < 10000; ++epoch) {
    const Tensor prediction = network.forward(inputs);
    network.backward(loss.backward(prediction, targets));
    network.update(0.05);
}
```

Include `Tensor.hpp`, `NeuralNetwork.hpp`, `Linear.hpp`, `ReLU.hpp`, and
`MSELoss.hpp` when using this snippet. Each epoch processes the complete dataset.
The scalar `loss.forward()` is useful for reporting; parameter updates need the
derivative from `loss.backward()`, so the example computes scalar loss only before
and after training.

No layer contains an XOR rule. Training adjusts parameters so the learned
function matches the supplied targets. To see why the architecture can represent
XOR, consider this possible solution for binary inputs:

```text
XOR(x₁, x₂) = ReLU(x₁ - x₂) + ReLU(x₂ - x₁)
```

| Inputs | First term | Second term | Sum |
| --- | --- | --- | --- |
| `0, 0` | 0 | 0 | 0 |
| `0, 1` | 0 | 1 | 1 |
| `1, 0` | 1 | 0 | 1 |
| `1, 1` | 0 | 0 | 0 |

Two hidden neurons can represent this expression. The example uses eight to give
training more flexibility and may learn a different solution with the same four
answers. The output layer has no activation: its values approximate zero and one
but are not constrained probabilities.

Seeds 42 and 43 make initialization repeatable with the same standard library.
Random distribution implementations can differ across libraries; exact losses
and predictions can vary. A successful run prints predictions close to
`[0, 1, 1, 0]` and `XOR passed`.

## Tests and what they establish

The tests use plain C++ and the standard library. Google Test and Catch2 are not
used. CTest is the runner: it launches executables and interprets their exit codes.

| CTest name | Executable | Checks |
| --- | --- | --- |
| `components` | `cpp_ml_tests` | Tensor operations, selected invalid inputs, activation behavior, numerical gradients, and a loss-reducing update |
| `xor` | `cpp_ml_example` | Complete network training and all four predictions |

Despite its name, `tests/TensorTests.cpp` also tests layers and loss. Its `near()`
helper rejects nonfinite values and values outside a tolerance. Its `throws()`
helper checks expected exceptions. Failures return a nonzero exit code; checks
do not rely on `assert`, so they remain active in optimized builds.

Gradient checks compare analytical derivatives with central finite differences:

```text
dL/dθ ≈ (L(θ + ε) - L(θ - ε)) / (2ε)
```

They cover MSE's prediction gradient, ReLU away from its nondifferentiable point,
and Linear's input, weight, and bias gradients. A separate check verifies a
small gradient descent step reduces loss.

The XOR executable succeeds only if final MSE is finite, lower than initial MSE,
and below `0.001`, with every prediction finite and less than `0.1` away from its
target. Learning XOR checks that the nonlinear forward path and training machinery
work together. It does not prove generalization beyond the four training examples
or replace broader operation and gradient tests.

## Future work

Potential extensions include sigmoid and softmax, cross-entropy loss, separate
optimizer classes, mini-batch training, regression examples, and broader tests.
General N-dimensional tensors, automatic differentiation, serialization, and
performance work can follow as the project grows. None are required for the
current XOR example.
