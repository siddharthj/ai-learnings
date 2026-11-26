# 2026-11-15: Karpathy, micrograd

**Source:** Andrej Karpathy, [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) (Zero to Hero, lecture 1)
**Notebook / code:** [`micro_grad.ipynb`](micro_grad.ipynb)

## What I did and the result
Finished the lecture and coded along in my own notebook: a scalar `Value` class that records how each number was computed, a backward pass that fills in every gradient, then a small neural network (`Neuron`, `Layer`, `MLP`) trained on four toy examples.

The notebook checks the engine against PyTorch: a single neuron gives output 0.7071, and the gradients of `x1`, `w1`, `x2` and `w2` come out as -1.5, 1.0, 0.5 and 0.0 in the saved output.

## How I did it
Typed the code in my own notebook while watching, pausing before each step. The `Value` class follows Karpathy's published [micrograd](https://github.com/karpathy/micrograd) engine.

## What confused me, and how I got unstuck
- **What micrograd and a loss function are.** Micrograd is a tiny engine that tracks how a number was computed so it can work out, for every input, how much it changed the output. The loss is one number for how wrong the model is.
- **Is the gradient used in the forward pass, and is it the weight?** No to both. Weights are the model's numbers and the forward pass uses them. Gradients are computed afterwards, one per weight, and say which way and how much the loss changes if that weight is nudged.
- **How and when the weights change.** Once per step, after the backward pass: `weight = weight - learning_rate * gradient`. Worked through a one-weight example (w = 1, x = 2, target 10) until the sign made sense: a negative gradient means raising the weight lowers the loss.
- **A typo that stopped the notebook.** The backward pass called `v.c()` instead of `v._backward()`, so the first `o.backward()` raised an `AttributeError` and nothing after it ran. Found while reviewing the notebook.

## What I learned
- Backpropagation is the chain rule applied node by node, walking the graph backwards from the output.
- Gradients are added with `+=`, not assigned, because a node used more than once receives gradient from every use. Assigning would silently drop all but one.
- Gradients have to be reset to zero between steps, or the next backward pass adds onto the old ones.
- Training is one loop repeated: forward, loss, backward, update. The same loop trains GPT; only the model and the loss change.
- Checking my own engine against a second implementation (the PyTorch cell) is how to know the gradients are right.

## What's next
Next lecture in the series. (Suggested: makemore part 1.)

## How to verify
Install `requirements.txt`, open `micro_grad.ipynb`, and choose Restart Kernel and Run All Cells. The graph-drawing cell also needs the Graphviz program (`brew install graphviz` on a Mac).

| Check | Expected |
|---|---|
| `f(3.0)` | 20.0 |
| Single neuron output `o` | 0.7071 |
| Gradients of `x1`, `w1`, `x2`, `w2` | -1.5, 1.0, 0.5, 0.0, matching the PyTorch cell |
| Training loop (30 steps, learning rate 0.01) | loss falls over the steps. Weights start random, so exact numbers vary. |

The `Value` class follows Karpathy's [micrograd](https://github.com/karpathy/micrograd) engine. A typo in the backward pass (`v.c()` instead of `v._backward()`) stopped the notebook after the first backward call, and it is fixed. Short comments were added and no other code was changed.
