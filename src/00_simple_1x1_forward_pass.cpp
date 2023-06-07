#include <torch/torch.h>

int main()
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

    // Print the output
    std::cout << "Output: " << output << std::endl;

    return 0;
};
