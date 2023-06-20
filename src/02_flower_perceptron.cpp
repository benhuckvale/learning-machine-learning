#include <torch/torch.h>
#undef CHECK // Undefine conflicting macro
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

torch::nn::Linear make_flower_perceptron()
{
    // Define the input data (the training data)
    // A flower of class 0 with petal length 0.5, width 0.3
    // A flower of class 1 with petal length 0.9, width 0.1
    torch::Tensor training_data{torch::tensor({{0.5, 0.3}, {0.9, 0.1}})};
    torch::Tensor training_labels{torch::tensor({{0}, {1}}).to(torch::kFloat)};

    // Define the neural network (perceptron)
    torch::nn::Linear linear(2, 1);
    torch::nn::Sigmoid activation;

    torch::nn::BCELoss loss_function;

    // Set the initial weight matrix and bias values (rather than allow for a random default).
    torch::nn::init::constant_(linear->weight, 0.5);
    torch::nn::init::constant_(linear->bias, 0.5);

    torch::optim::SGD optimizer(linear->parameters(), torch::optim::SGDOptions(0.1));

    // Perform the training
    for (int epoch = 0; epoch < 1000; ++epoch) {
        // Forward pass
        torch::Tensor output = activation(linear->forward(training_data));

        // Compute the loss
        torch::Tensor loss = loss_function(output, training_labels);

        // Backward pass (compute gradients)
        optimizer.zero_grad();
        loss.backward();

        // Update the weights and biases
        optimizer.step();

        // Print the current state of the weights and biases
        //auto weight = linear->weight.data();
        //auto bias = linear->bias.data();
        //std::cout << "Epoch: " << epoch << ", Weight: " << weight << ", Bias: " << bias << std::endl;
    }
    return linear;
}

double predict(torch::nn::Linear& trained_nn, double length, double width)
{
    torch::nn::Sigmoid activation;
    torch::Tensor flower{torch::tensor({length, width})};
    torch::Tensor prediction{activation(trained_nn->forward(flower))};
    return prediction.item<double>();
}

TEST_CASE("Test Flower Perceptron")
{
    GIVEN("A trained NN") {
        auto trained_nn{make_flower_perceptron()};

        WHEN("I make a prediction for something that ought to be a class 0 flower") {
            const double prediction{predict(trained_nn, 0.4, 0.4)};

            THEN("It has the expected value") {
                CHECK(prediction < 0.5);
                CHECK(prediction == doctest::Approx(0.1153).epsilon(0.0001));
            }
        }

        WHEN("I make a prediction for something that ought to be a class 1 flower") {
            const double prediction{predict(trained_nn, 0.8, 0.2)};

            THEN("It has the expected value") {
                CHECK(prediction > 0.5);
                CHECK(prediction == doctest::Approx(0.6426).epsilon(0.0001));
            }
        }
    }
}

int main(int argc, char* argv[])
{
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    if (context.shouldExit()) return context.run();

    auto trained_nn{make_flower_perceptron()};
    std::cout << "0.6, 0.3 => "<<predict(trained_nn, 0.6, 0.3)<<std::endl;
    std::cout << "0.7, 0.2 => "<<predict(trained_nn, 0.7, 0.2)<<std::endl;
    std::cout << "0.8, 0.1 => "<<predict(trained_nn, 0.8, 0.1)<<std::endl;
    return 0;
}
