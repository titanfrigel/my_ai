/*
** titanfrigel
** my_ai
** File description:
** my_neural_network.h
*/

#include "my_matrix.h"

#ifndef _MY_NEURAL_NETWORK_H_
    #define _MY_NEURAL_NETWORK_H_
    template <typename Type>
    class my_neural_network
    {
        public:
        std::vector<size_t> units_per_layer;
        std::vector<my_matrix<Type>> weight_matrices;
        std::vector<my_matrix<Type>> bias_vectors;
        std::vector<my_matrix<Type>> activations;

        float lr;

        explicit my_neural_network(std::vector<size_t> units_per_layer, float lr) :
		units_per_layer(units_per_layer), weight_matrices(), bias_vectors(),
		activations(), lr(lr)
		{
			for (size_t i = 0; i < units_per_layer.size() - 1; ++i) {
				size_t in_channels{units_per_layer[i]};
				size_t out_channels{units_per_layer[i + 1]};

				auto W = mtx<Type>::randn(out_channels, in_channels);
				weight_matrices.push_back(W);

				auto b = mtx<Type>::randn(out_channels, 1);
				bias_vectors.push_back(b);

				activations.resize(units_per_layer.size());
			}
		}

        static inline auto sigmoid(float x)
		{
			return (1.0f / (1 + exp(-x)));
		}

        static inline auto d_sigmoid(float x)
		{
			return (sigmoid(x) * (1 - sigmoid(x)));
		}

        auto forward(my_matrix<Type> x)
        {
			assert(x.rows == units_per_layer[0] && x.cols);

			activations[0] = x;
			my_matrix prev(x);
			for (size_t i = 0; i < units_per_layer.size() - 1; ++i) {
				my_matrix y = weight_matrices[i].matmul(prev);
				y = y + bias_vectors[i];
				y = y.apply_function(sigmoid);
				activations[i + 1] = y;
				prev = y;
			}
			return (prev);
        }

        void backprop(my_matrix<Type> target)
		{
			assert(target.rows == units_per_layer.back());

			auto y_hat = activations.back();
			auto error = (target - y_hat).square();

			for (int i = weight_matrices.size() - 1; i >= 0; --i) {
				auto Wt = weight_matrices[i].T();
				auto prev_errors = Wt.matmul(error);

				auto d_outputs = activations[i + 1].apply_function(d_sigmoid);
				auto gradients = error.multiply_elementwise(d_outputs);
				gradients = gradients.multiply_scalar(lr);

				auto a_trans = activations[i].T();
				auto weight_gradient = gradients.matmul(a_trans);


				bias_vectors[i] = bias_vectors[i].add(gradients);
				weight_matrices[i] = weight_matrices[i].add(weight_gradient);
				error = prev_errors;
			}
		}
    };
#endif
