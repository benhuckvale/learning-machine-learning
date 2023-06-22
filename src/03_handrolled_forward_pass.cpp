#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <cmath>
#include <numeric>
#include <vector>

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double forward(const std::vector<double>& input, const std::vector<double>& weight, double bias) {
    double sum = std::inner_product(input.begin(), input.end(), weight.begin(), bias);
    return sigmoid(sum);
}

TEST_CASE("Neural Network Layer") {
    SUBCASE("Forward pass on some arbitrary values") {
        std::vector<double> input{0.5, 0.3};
        std::vector<double> weight{2.0, 1.0};
        double bias{0.5};

        double result{forward(input, weight, bias)};
        CHECK(result == doctest::Approx(0.818731));
    }

    SUBCASE("Forward pass when weighted in favour of input") {
        std::vector<double> input{1.0, 0.0};
        std::vector<double> weight{1.0, 0.0};
        double bias{0.0};

        double result{forward(input, weight, bias)};
        CHECK(result == doctest::Approx(0.7310585786));
    }

    SUBCASE("Forward pass when weighted against input") {
        std::vector<double> input{1.0, 0.0};
        std::vector<double> weight{0.0, 1.0};
        double bias{0.0};

        double result{forward(input, weight, bias)};
        CHECK(result == doctest::Approx(0.5));
    }
}

int main(int argc, char* argv[])
{
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    return context.run();
}
