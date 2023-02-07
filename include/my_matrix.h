/*
** titanfrigel
** my_ai
** File description:
** my_matrix.h
*/

#include <vector>
#include <iostream>
#include <random>
#include <cassert>
#include <cmath>
#include <functional>

#ifndef _MY_MATRIX_H_
    #define _MY_MATRIX_H_
    template <typename Type>
    class my_matrix
    {
        public:
        size_t cols;
        size_t rows;
        std::vector<Type> data;

        my_matrix(size_t rows, size_t cols) : cols(cols), rows(rows), data({})
        {
            data.resize(cols * rows, Type());
        }

        my_matrix() : cols(0), rows(0), data({}) {};

        void print_shape()
        {
            std::cout << "Matrix Size([" << rows << ", " << cols << "])" << std::endl;
        }

        void print()
        {
            for (size_t r = 0; r < rows; r++) {
                for (size_t c = 0; c < cols; c++) {
                    std::cout << (*this)(r, c) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }

        Type& operator()(size_t row, size_t col) {
            return data[row * cols + col];
        }

        my_matrix matmul(my_matrix &target)
        {
            assert(cols == target.rows);
            my_matrix output(rows, target.cols);

            for (size_t r = 0; r < output.rows; ++r)
                for (size_t c = 0; c < output.cols; ++c)
                    for (size_t k = 0; k < target.rows; ++k)
                        output(r, c) += (*this)(r, k) * target(k, c);
            return (output);
        }

        my_matrix multiply_elementwise(my_matrix &target)
        {
            assert(cols == target.cols && rows == target.rows);
            my_matrix output((*this));

            for (size_t r = 0; r < output.rows; ++r)
                for (size_t c = 0; c < output.cols; ++c)
                    output(r, c) += (*this)(r, c) * target(r, c);
            return (output);
        }

        my_matrix square()
        {
            my_matrix output((*this));
            output = multiply_elementwise(output);
            return (output);
        }

        my_matrix multiply_scalar(Type scalar)
        {
            my_matrix output((*this));
            for (size_t r = 0; r < output.rows; ++r)
                for (size_t c = 0; c < output.cols; ++c)
                    output(r, c) = scalar * (*this)(r, c);
            return (output);
        }

        my_matrix add(my_matrix &target)
        {
            assert(cols == target.cols && rows == target.rows);
            my_matrix output(rows, target.cols);

            for (size_t r = 0; r < output.rows; ++r)
                for (size_t c = 0; c < output.cols; ++c)
                    output(r, c) = (*this)(r, c) + target(r, c);
            return (output);
        }

        my_matrix operator+(my_matrix &target)
        {
            return (add(target));
        }

        my_matrix operator-()
        {
            my_matrix output(rows, cols);

            for (size_t r = 0; r < output.rows; ++r)
                for (size_t c = 0; c < output.cols; ++c)
                    output(r, c) = -(*this)(r, c);
            return (output);
        }

        my_matrix sub(my_matrix &target)
        {
            my_matrix neg_target = -target;
            return (add(neg_target));
        }

        my_matrix operator-(my_matrix &target)
        {
            return (sub(target));
        }

        my_matrix transpose()
        {
            size_t new_rows(cols), new_cols(rows);
            my_matrix transposed(new_rows, new_cols);

            for (size_t r = 0; r < new_rows; ++r)
                for (size_t c = 0; c < new_cols; ++c)
                    transposed(r, c) = (*this)(c, r);
            return (transposed);
        }

        my_matrix T()
        {
            return (transpose());
        }

        my_matrix apply_function(const std::function<Type(const Type &)> &function)
        {
            my_matrix output((*this));

            for (size_t r = 0; r < rows; ++r)
                for (size_t c = 0; c < cols; ++c)
                    output(r, c) = (function)((*this)(r, c));
            return (output);
        }

        my_matrix kronecker(my_matrix &target)
        {
            my_matrix output(target.rows, cols);

            for (size_t r = 0; r < target.rows; ++r)
                for (size_t c = 0; c < cols; ++c)
                    output(r, c) = (*this)(0, c) * target(r, 0);
            return (output);
        }

        my_matrix concatenate(my_matrix &target)
        {
            assert(rows == target.rows);

            my_matrix output(rows, cols + target.cols);
            for (size_t r = 0; r < rows; ++r)
                for (size_t c = 0; c < cols; ++r)
                    output(r, c) = (*this)(r, c);
            for (size_t r = 0; r < target.rows; ++r)
                for (size_t c = 0; c < target.cols; ++r)
                    output(r, c + cols) = target(r, c);
            return (output);
        }
    };
    template <typename Type>
    struct mtx {
        static my_matrix<Type> randn(size_t rows, size_t cols) {
            my_matrix<Type> M(rows, cols);
            std::random_device rd{};
            std::mt19937 gen{rd()};
            Type n(11);
            Type stdev{(Type)(1 / sqrt(n))};
            std::normal_distribution<Type> d{0, stdev};
            for (size_t r = 0; r < rows; ++r)
                for (size_t c = 0; c < cols; ++c)
                    M(r, c) = d(gen);
        return M;
        }
    };
#endif
