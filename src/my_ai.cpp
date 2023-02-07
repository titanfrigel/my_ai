/*
** titanfrigel
** my_ai
** File description:
** my_ai.cpp
*/

// #include "my.h"
// #include "nn.h"
// #include "gfk.h"
// #include <fstream>

#include "my_neural_network.h"

auto make_model(size_t in_channels, size_t out_channels,
size_t hidden_units_per_layers, int hidden_layers, float lr)
{
	std::vector<size_t> units_per_layer;
	units_per_layer.push_back(in_channels);
	for (int i = 0; i < hidden_layers; ++i) {
		units_per_layer.push_back(hidden_units_per_layers);
	}
	units_per_layer.push_back(out_channels);
	my_neural_network<float> model(units_per_layer, lr);
	return (model);
}

int my_ai(int argc, char **argv)
{
	size_t in_channels = 1;
	size_t out_channels = 1;
	size_t hidden_units_per_layers = 2;
	size_t hidden_layers = 2;
	float lr = .05f;

	auto model = make_model(in_channels, out_channels,
	hidden_units_per_layers, hidden_layers, lr);

	size_t iterations = 100000000;

	double loseprinter = 0;
	double loseprinter_perc = 0;
	double losesave = 0;
	float percentage = 0;
	printf("%.1f%%\n", percentage);
	for (size_t i = 0; i < iterations; ++i) {
		auto x = mtx<float>::randn(in_channels, 1);
		my_matrix<float> y(out_channels, 1);
		y(0, 0) = sin(x(0, 0)) * sin(x(0, 0));

		auto y_hat = model.forward(x);

		model.backprop(y);

		loseprinter_perc = (loseprinter_perc + pow(y.sub(y_hat)(0, 0) , 2)) / 2;
		loseprinter = (loseprinter + pow(y.sub(y_hat)(0, 0), 2)) / 2;
		if (i % (iterations / 1000) == 0) {
			printf("Lose at %.1f%%: %f\n", percentage += 0.1, loseprinter_perc);
			loseprinter_perc = 0;
		}
		if (i > iterations - 10) {
			printf("New:\n");
			x.print();
			y.print();
			y_hat.print();
		}
		if (i == iterations / 100)
			losesave = loseprinter;
	}
	for (size_t i = 0; i < model.weight_matrices.size(); ++i) {
		model.weight_matrices[i].print();
		model.bias_vectors[i].print();
	}
	printf("\nLose at 1%% (%lu): %f\nGlobal lose: %f\n", iterations, losesave, loseprinter);

	return (argc + argv[0][0]);
}
