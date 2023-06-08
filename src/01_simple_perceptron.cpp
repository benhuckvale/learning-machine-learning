#include <torch/torch.h>
#undef CHECK // Undefine conflicting macro
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

double simple_perceptron()
{
    // Define the input data
    const torch::Tensor input{torch::ones({1, 2})};

    // Define the neural network (perceptron)
    torch::nn::Linear linear(2, 1);
    torch::nn::Sigmoid activation;

    // Set the initial weight matrix and bias values (rather than allow for a random default).
    // Use init::constant_ to fill the weight matrix with 0.5, to become: {0.5, 0.5}
    torch::nn::init::constant_(linear->weight, 0.5);
    torch::nn::init::constant_(linear->bias, 0.5);

    // Perform a forward pass
    const torch::Tensor output{activation(linear->forward(input))};

    return output.item<double>();
}

TEST_CASE("Simple Perceptron")
{
    // Test against the known definition of sigmoid
    const auto sigmoid{[](const double& input) { return 1.0 / (1.0 + std::exp(-input)); }};

    // Calculate sigmoid of expected sum of weighted inputs + bias
    const auto expected{sigmoid(0.5*1 + 0.5*1 + 0.5)};

    CHECK(simple_perceptron() == doctest::Approx(expected));
}

int main(int argc, char* argv[])
{
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    if (context.shouldExit()) return context.run();

    const auto output{simple_perceptron()};
    std::cout << "Output: " << output << std::endl;
    return 0;
}
