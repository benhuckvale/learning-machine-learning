#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

// Operator overloads for vector operations
template<typename T>
std::vector<T> operator+(const std::vector<T>& lhs, const std::vector<T>& rhs) {
    std::vector<T> result;
    std::transform(lhs.begin(), lhs.end(), rhs.begin(), std::back_inserter(result), std::plus<T>());
    return result;
}

template<typename T>
std::vector<T> operator-(const std::vector<T>& lhs, const std::vector<T>& rhs) {
    std::vector<T> result;
    std::transform(lhs.begin(), lhs.end(), rhs.begin(), std::back_inserter(result), std::minus<T>());
    return result;
}

template<typename T>
std::vector<T> operator*(const std::vector<T>& lhs, const std::vector<T>& rhs) {
    std::vector<T> result;
    std::transform(lhs.begin(), lhs.end(), rhs.begin(), std::back_inserter(result), std::multiplies<T>());
    return result;
}

template<typename T>
std::vector<T> operator*(const std::vector<T>& vec, const T& scalar) {
    std::vector<T> result;
    std::transform(vec.begin(), vec.end(), std::back_inserter(result), [scalar](const T& element) {
        return element * scalar;
    });
    return result;
}

template<typename T>
std::vector<T> operator*(const T& scalar, const std::vector<T>& vec) {
    return vec * scalar;
}

template<typename T>
std::vector<T> operator-(const std::vector<T>& vec, const T& lower_rank) {
    std::vector<T> result;
    std::transform(vec.begin(), vec.end(), std::back_inserter(result), [&lower_rank](const T& element) {
        return element - lower_rank;
    });
    return result;
}

template<typename T>
std::vector<T> operator-(const T& lower_rank, const std::vector<T>& vec) {
    std::vector<T> result;
    std::transform(vec.begin(), vec.end(), std::back_inserter(result), [&lower_rank](const T& element) {
        return lower_rank - element;
    });
    return result;
}

double dot(const std::vector<double>& a, const std::vector<double>& b)
{
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

// Sigmoid function
template<typename T>
T sigmoid(const T& x) {
    return 1 / (1 + std::exp(-x));
}

std::vector<double> make_single_neuron(
    int input_counts,
    std::vector<std::vector<double>> training_inputs_sets,
    std::vector<double> training_outputs,
    int iteration_count = 100
)
{
    std::vector<double> weights(input_counts, 0.2);

    // Perform the training
    for (int iteration{0}; iteration < iteration_count; ++iteration) {
        // Forward pass
        std::vector<double> output;
        std::transform(
            training_inputs_sets.cbegin(),
            training_inputs_sets.cend(),
            std::back_inserter(output),
            [&weights](const std::vector<double>& inputs) {
                const auto dot_product{dot(inputs, weights)};
                const double sigmoid_output{sigmoid(dot_product)};
                return sigmoid_output;
            }
        );

        // Compute the error
        std::vector<double> errors{training_outputs - output};

        // Compute the gradients
        std::vector<double> gradients{output * (1.0 - output) * errors};

        // Update the weights
        for(int i{0}; i<weights.size(); ++i) {
            const auto update{std::transform_reduce(
                training_inputs_sets.cbegin(),
                training_inputs_sets.cend(),
                gradients.cbegin(),
                0.0,
                std::plus<double>{},
                [i](const std::vector<double>& inputs, const double gradient) {
                    return inputs[i]*gradient;
                }
            )};
            weights[i] += update;
        }
    }
    return weights;
}
 
TEST_CASE("Neuron with 3 inputs") {
    // The scenario is that each input has 3 characteristics but the second and
    // third are just noise. Let's build a single neuron that can see through that.
    // What should happen is that at the end of the training the weights on the
    // latter inputs should be close to zero.

    // Training input.
    std::vector<std::vector<double>> training_inputs_sets{
        {0, 0, 1}, {1, 1, 1}, {1, 0, 1}, {0, 1, 1}
    };
    std::vector<double> training_outputs{0, 1, 1, 0};

    auto neuron{make_single_neuron(3, training_inputs_sets, training_outputs)};

    SUBCASE("Positive case, not seen before") {
        std::vector<double> test_input = {1, 0, 0};
        double test_output{sigmoid(dot(test_input, neuron))};
        CHECK(test_output == doctest::Approx(0.99).epsilon(0.01));
    }

    SUBCASE("Negative case, seen in training") {
        std::vector<double> test_input = {0, 0, 1};
        double test_output{sigmoid(dot(test_input, neuron))};
        CHECK(test_output == doctest::Approx(0.11).epsilon(0.01));
    }

    SUBCASE("Negative case, not seen in training") {
        std::vector<double> test_input = {0, 1, 0};
        double test_output{sigmoid(dot(test_input, neuron))};
        CHECK(test_output == doctest::Approx(0.44).epsilon(0.01));
    }
}

int main(int argc, char* argv[])
{
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    return context.run();
}
