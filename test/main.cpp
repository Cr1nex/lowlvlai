#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

//Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1 - x);
}

// Layer class single layer 
class Layer {
public:
    int input_size;
    int output_size;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    std::vector<double> outputs;
    std::vector<double> inputs;
    std::vector<double> deltas;

    Layer(int input_size, int output_size) : input_size(input_size), output_size(output_size) {
        // Initialize weights and biases randomly
        srand(time(0));

        weights.resize(input_size, std::vector<double>(output_size));
        biases.resize(output_size);
        outputs.resize(output_size);
        deltas.resize(output_size);

        for (int i = 0; i < input_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                weights[i][j] = (rand() % 1000) / 1000.0 - 0.5;  // Random weights between -0.5 and 0.5
            }
        }

        for (int i = 0; i < output_size; ++i) {
            biases[i] = (rand() % 1000) / 1000.0 - 0.5;  // Random biases between -0.5 and 0.5
        }
    }

    // Forward
    std::vector<double> forward(const std::vector<double>& input) {
        inputs = input;
        for (int i = 0; i < output_size; ++i) {
            outputs[i] = 0;
            for (int j = 0; j < input_size; ++j) {
                outputs[i] += inputs[j] * weights[j][i];
            }
            outputs[i] += biases[i];
            outputs[i] = sigmoid(outputs[i]);
        }
        return outputs;
    }

    // Backward
    std::vector<double> backward(const std::vector<double>& error, double learning_rate) {
        for (int i = 0; i < output_size; ++i) {
            deltas[i] = error[i] * sigmoid_derivative(outputs[i]);
        }

        std::vector<double> prev_error(input_size, 0);
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < input_size; ++j) {
                weights[j][i] -= learning_rate * deltas[i] * inputs[j];
            }
            biases[i] -= learning_rate * deltas[i];
        }

        for (int i = 0; i < input_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                prev_error[i] += deltas[j] * weights[i][j];
            }
        }

        return prev_error;
    }
};

//  The network
class MLP {
public:
    std::vector<Layer> layers;

    MLP(const std::vector<int>& layer_sizes) {
        for (int i = 1; i < layer_sizes.size(); ++i) {
            layers.push_back(Layer(layer_sizes[i-1], layer_sizes[i]));
        }
    }

    // Forward pass all layers
    std::vector<double> forward(const std::vector<double>& input) {
        std::vector<double> current_input = input;
        for (auto& layer : layers) {
            current_input = layer.forward(current_input);
        }
        return current_input;
    }

    // Backward pass all layers
    void backward(const std::vector<double>& X, const std::vector<double>& y, double learning_rate) {
        std::vector<double> output = forward(X);
        std::vector<double> error(y.size());
        
        // Calculate error at the output layer
        for (int i = 0; i < y.size(); ++i) {
            error[i] = output[i] - y[i];
        }

        // Propagate the error back through each layer
        for (int i = layers.size() - 1; i >= 0; --i) {
            error = layers[i].backward(error, learning_rate);
        }
    }

    // Train network
    void train(const std::vector<std::vector<double>>& X_train, const std::vector<std::vector<double>>& y_train,
               int epochs, double learning_rate) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (int i = 0; i < X_train.size(); ++i) {
                backward(X_train[i], y_train[i], learning_rate);
            }

            if (epoch % 100 == 0) {
                double loss = 0;
                for (int i = 0; i < X_train.size(); ++i) {
                    std::vector<double> output = forward(X_train[i]);
                    for (int j = 0; j < y_train[i].size(); ++j) {
                        loss += (output[j] - y_train[i][j]) * (output[j] - y_train[i][j]);
                    }
                }
                loss /= X_train.size();
                std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
            }
        }
    }
};

int main() {
    // Network architecture
    std::vector<int> layer_sizes = {3, 4, 1}; // 3 inputs, 4 hidden neurons, 1 output neuron
    MLP mlp(layer_sizes);

    // Training data (XOR problem example)
    std::vector<std::vector<double>> X_train = {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, 
                                               {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}};
    std::vector<std::vector<double>> y_train = {{0}, {1}, {1}, {0}, {1}, {0}, {0}, {1}};

    // Declare epochs and learning rate
    int epochs = 10000;
    double learning_rate = 0.1;

    // Train the model
    mlp.train(X_train, y_train, epochs, learning_rate);

    // Test the model
    for (const auto& input : X_train) {
        std::vector<double> output = mlp.forward(input);
        std::cout << "Input: ";
        for (double val : input) std::cout << val << " ";
        std::cout << "Output: " << output[0] << std::endl;
    }

    return 0;
}
