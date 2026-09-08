#include "Tensor.hpp"

#include <iomanip>
#include <ostream>
#include <stdexcept>

// ctor
Tensor::Tensor(std::size_t rows, std::size_t cols)
    : rows_{rows},
      cols_{cols},
      data_(rows * cols, 0.0)
{
}

Tensor::Tensor(
    std::initializer_list<
        std::initializer_list<double>> values)
    : rows_{values.size()},
      cols_{values.size() == 0
                ? 0
                : values.begin()->size()}
{
    data_.reserve(rows_ * cols_);

    for (const auto& row : values) {
        if (row.size() != cols_) {
            throw std::invalid_argument{
                "Tensor rows must have equal length"
            };
        }

        data_.insert(data_.end(),
                     row.begin(),
                     row.end());
    }
}

// element access
double& Tensor::operator()(std::size_t row,
                           std::size_t col)
{
    return data_[index(row, col)];
}

const double& Tensor::operator()(
    std::size_t row,
    std::size_t col) const
{
    return data_.at(index(row, col));
}

// Shape methods
std::size_t Tensor::rows() const noexcept
{
    return rows_;
}

std::size_t Tensor::cols() const noexcept
{
    return cols_;
}

std::size_t Tensor::size() const noexcept
{
    return data_.size();
}

// index calc
std::size_t Tensor::index(
    std::size_t row,
    std::size_t col) const
{
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range{
            "Tensor index out of bounds"
        };
    }

    return row * cols_ + col;
}

// transpose
Tensor Tensor::transpose() const
{
    Tensor result{cols_, rows_};

    for (std::size_t r = 0; r < rows_; ++r) {
        for (std::size_t c = 0; c < cols_; ++c) {
            result(c, r) = (*this)(r, c);
        }
    }

    return result;
}

// shape validation
void Tensor::check_same_shape(
    const Tensor& rhs) const
{
    if (rows_ != rhs.rows_ ||
        cols_ != rhs.cols_) {
        throw std::invalid_argument{
            "Tensor shape mismatch"
        };
    }
}

// operator overloads
Tensor& Tensor::operator+=(const Tensor& rhs)
{
    check_same_shape(rhs);

    for (std::size_t i = 0;
         i < data_.size();
         ++i) {
        data_[i] += rhs.data_[i];
    }

    return *this;
}

Tensor& Tensor::operator-=(const Tensor& rhs)
{
    check_same_shape(rhs);

    for (std::size_t i = 0;
         i < data_.size();
         ++i) {
        data_[i] -= rhs.data_[i];
    }

    return *this;
}

Tensor& Tensor::operator*=(double scalar)
{
    for (double& value : data_) {
        value *= scalar;
    }

    return *this;
}

Tensor operator+(Tensor lhs,
                 const Tensor& rhs)
{
    lhs += rhs;
    return lhs;
}

Tensor operator-(Tensor lhs,
                 const Tensor& rhs)
{
    lhs -= rhs;
    return lhs;
}

Tensor operator*(Tensor tensor,
                 double scalar)
{
    tensor *= scalar;
    return tensor;
}

Tensor operator*(double scalar,
                 Tensor tensor)
{
    tensor *= scalar;
    return tensor;
}

// matmul
Tensor matmul(const Tensor& lhs,
              const Tensor& rhs)
{
    if (lhs.cols() != rhs.rows()) {
        throw std::invalid_argument{
            "Invalid dimensions for matrix multiplication"
        };
    }

    Tensor result{lhs.rows(), rhs.cols()};

    for (std::size_t i = 0;
         i < lhs.rows();
         ++i) {

        for (std::size_t j = 0;
             j < rhs.cols();
             ++j) {

            double sum = 0.0;

            for (std::size_t k = 0;
                 k < lhs.cols();
                 ++k) {

                sum += lhs(i, k) * rhs(k, j);
            }

            result(i, j) = sum;
        }
    }

    return result;
}

// for debugging
std::ostream& operator<<(std::ostream& os,
                         const Tensor& tensor)
{
    for (std::size_t r = 0; r < tensor.rows(); ++r) {

        os << "[ ";

        for (std::size_t c = 0; c < tensor.cols(); ++c) {
            os << std::setw(10)
               << tensor(r, c)
               << ' ';
        }

        os << "]\n";
    }

    return os;
}

Tensor Tensor::sum_rows() const
{
    Tensor result{1, cols_};

    for (std::size_t r = 0; r < rows_; ++r) {
        for (std::size_t c = 0; c < cols_; ++c) {
            result(0, c) += (*this)(r, c);
        }
    }

    return result;
}

void Tensor::add_row_vector(const Tensor& row)
{
    if (row.rows() != 1 ||
        row.cols() != cols_) {
        throw std::invalid_argument{
            "Invalid row vector shape"
        };
    }

    for (std::size_t r = 0; r < rows_; ++r) {
        for (std::size_t c = 0; c < cols_; ++c) {
            (*this)(r, c) += row(0, c);
        }
    }
}
