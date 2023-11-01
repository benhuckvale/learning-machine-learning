Learning Machine Learning
=========================

Welcome to the learning-machine-learning repository! What I'm trying to do here
is learn about machine-learning from the low-level, using really simple
examples.

Although I've done quite a bit of statistics in the past which puts me in a
good position, I lately thought to myself that I'm not a machine learning
expert, at least not at the time of the conception of this project, and perhaps
ought to be. So here I'm going to use that difficulty and ask the simplest and
dumbest questions (to a machine, e.g.  search engine/chat bot/..., or indeed, a
person) since I'm presently well-placed to do so, at least until I've learnt
too much, and refine and bottle up the answers in this project.

Despite using AI code-generation to assist, it is pretty difficult to get a
high quality coded example answer to something in just the right format even
with quite aggressive prompt-engineering, so rest-assured quite a few forward
passes of my own brain have gone into this project, mainly because I like
*tests*.  Everything here is a product of Human+Machine. That's the way to work
and do business in the 21st century.

Mostly this is C++20 and beyond. C++ might seem like an unconvential place to
start this learning journey I'm on, since Python is more commonly associated
with machine learning due to the rich ecosystem of libraries available. But
these libraries wrap details that I want to open up. And although here I have
examples using PyTorch (as libtorch) these are just for context and comparison
since primarily I want to do stuff with simple, even naive, code with no
dependencies so that hopefully the fundamentals are brought to light.
Ultimately this language choice is because I want to ground my
understanding of machine learning algorithms at the low-level of C++, gain
insight into the technical challenges of implementing them efficiently on CPU
and GPU architectures, and become able to build boutique AI solutions to things
within existing C++ projects. Plus, C++ is awesome.

I presented a short example from this project as a [lighting talk at the
2023 C++-on-sea conference](https://www.youtube.com/watch?v=0gYE5p7AXKw).

## Building and running

The prerequisite is that you need [cmake](https://cmake.org/download/)
installed. Then in your shell you should be able to:
```
cmake -B build -S .
cmake --build build
```

And then you can run things, for example:
```
build/00_simple_1x1_forward_pass
```

## Descriptions of each example
 
### Simplest possible forward pass

Uses libtorch.

What is a really simple program I can write that uses libtorch? Well calling
the forward pass function seems to be simple enough. I don't really care what
'forward pass' means at this point. I just want to demonstrate I can compile a
program that uses libtorch.

[00_simple_1x1_forward_pass.cpp](src/00_simple_1x1_forward_pass.cpp)

### Simple Perceptron

Uses libtorch.

A perceptron is about the simplest possible neural network one can create that
does anything useful.

[01_simple_perceptron.cpp](src/01_simple_perceptron.cpp)

### Simple flower perceptron

Uses libtorch.

The [Iris flower data set](https://en.wikipedia.org/wiki/Iris_flower_data_set)
is a particularly famous dataset, which is used now very much to teach data
science. We'll take that as inspiration and do something simple, by pretending
we have a few flowers (i.e. no more than 5 flowers) with different petal
lengths and widths and try to classify them.

[02_flower_perceptron.cpp](src/02_flower_perceptron.cpp)

### Single Neuron Example

No dependencies.

I found some good examples elsewhere that people had written, often in python,
to show a single neuron running off 3 inputs and producing 1 output. So I
translated the example into C++ in my own style, and wrote the functions needed
to train and use the model.

[06_handrolled_single_neuron](src/06_handrolled_single_neuron.cpp)

