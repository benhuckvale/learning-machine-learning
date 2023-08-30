Learning Machine Learning
=========================

Welcome to the learning-machine-learning repository! If you're reading this,
you're likely in the same boat as me – just starting out with machine learning
and perhaps also feeling a certain pressure to do so as it becomes inexorably
part of our world. If you are seasoned and experienced in the topic but
are seeking a new perspective I hope you find something of value.

I'm not a machine learning expert, at least not at the time of the conception
of this project. Instead, I'm an expert at being a beginner and I'm keen to
capture that and bottle it up in this project before I know too much, before I
lose that perspective, and before I lose the ability to ask the simplest and
stupidest questions.  I'm hoping that will mean this repository has a certain
flavour that will complement other material, courses and books one can follow
elsewhere. Very much it is my journey but I hope it is also useful to you.

I'm going to try to keep things really simple. For example, I'm frequently
going to address topics on a small scale if I can find a way to do so. I feel
this is not so well-trodden a path because in general the real world utility of
machine learning comes about because of models trained on large numbers of
complex items, with large numbers of connections.

Having said that, I'm a seasoned software developer, so I'm not going to hold
back on programming techniques to frame all of this work. We will have
a build system, we will have structure to our programs, and we will have unit
tests. I am going to use some existing libraries (PyTorch, as libtorch), but
I'm also going to write some things from scratch with no dependencies (except
the unit test library I'm using).

Mostly this is all going to be in C++20 and beyond. C++ might seem like an
unconvential place to start, since Python is more commonly associated with
machine learning due to the rich ecosystem of libraries available. But these
libraries wrap details that I want to open up, and as mentioned I want to do
things from scratch sometimes.  Ultimately this language choice is because I
personally want to ground my understanding of machine learning algorithms at
the low-level of C++ and gain insight into the technical challenges of
implementing them efficiently on CPU and GPU architectures.

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

