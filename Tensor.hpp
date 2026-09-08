#pragma once

#include <cstddef>
#include <initializer_list>
#include <iosfwd>
#include <vector>

class Tensor {
public:
    Tensor(std::size_t rows, std::size_t cols);

    Tensor(std::initializer_list<
        std::initializer_list<double>> values);

    double& operator()(std::size_t row, std::size_t col);
    const double& operator()(std::size_t row,
                             std::size_t col) const;

    // Shape
    [[nodiscard]] std::size_t rows() const noexcept;
    [[nodiscard]] std::size_t cols() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;

    // Basic operations
    [[nodiscard]] Tensor transpose() const;

    Tensor& operator+=(const Tensor& rhs);
    Tensor& operator-=(const Tensor& rhs);
    Tensor& operator*=(double scalar);

private:
    std::size_t rows_;
    std::size_t cols_;

    std::vector<double> data_;

    [[nodiscard]]
    std::size_t index(std::size_t row,
                      std::size_t col) const;

    void check_same_shape(const Tensor& rhs) const;
};

// Non-member operators
Tensor operator+(Tensor lhs, const Tensor& rhs);
Tensor operator-(Tensor lhs, const Tensor& rhs);
Tensor operator*(Tensor tensor, double scalar);
Tensor operator*(double scalar, Tensor tensor);

// Matrix multiplication
Tensor matmul(const Tensor& lhs, const Tensor& rhs);

// Debugging / printing
std::ostream& operator<<(std::ostream& os,
                         const Tensor& tensor);
}
