#include <torch/torch.h>
#undef CHECK // Undefine conflicting macro
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

double simple_forward_pass()
{
    // Define the input data
    // 'ones()' creates a tensor of dimension 1x1 filled with 1s. I.e. input is '1'
    const torch::Tensor input{torch::ones({1, 1})};

    // Define the neural network. A linear layer that performs a matrix
    // multiplication followed by an addition.
    // Input size is 1. Output size is // 1.
    torch::nn::Linear linear(1, 1);

    // Set the initial weight matrix and bias values (rather than allow for a random default).
    torch::nn::init::constant_(linear->weight, 2.0);
    torch::nn::init::constant_(linear->bias, 0.5);

    // Perform a forward pass
    const torch::Tensor output{linear->forward(input)};

    // Forward pass is:
    // output = input * weight + bias
    //        = 1 * 2.0 + 0.5
    //        = 2.5

    return output.item<double>();
}

TEST_CASE("Simple Forward Pass")
{
    CHECK(simple_forward_pass() == doctest::Approx(2.5));
}

int main(int argc, char* argv[])
{
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    if (context.shouldExit()) return context.run();

    auto output{simple_forward_pass()};
    std::cout << "Output: " << output << std::endl;
    return 0;
};
