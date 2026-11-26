# Learning how LLMs are built, from scratch

Notebooks and notes from working through Andrej Karpathy's [Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero) series. Each project has its own folder with the notebook, requirements, a journal (what I did, what broke, how to check it) and learnings (interview-style questions and answers in plain English), so any one of them can be checked on its own.

**Credit:** the lectures, the micrograd engine and the `names.txt` data set are Andrej Karpathy's ([nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero), [micrograd](https://github.com/karpathy/micrograd), [makemore](https://github.com/karpathy/makemore)). The notebooks are code I typed while following the lectures, with my own comments and fixes. The journals and learnings are my own notes.

## Projects

| # | Project | Finished | What it shows | Journal | Learnings | Notebook |
|---|---|---|---|---|---|---|
| 01 | Intro to LLMs and Deep Dive into LLMs | 2026-11-11 | Watched, no code | [journal](01-intro-videos/journal.md) | n/a | n/a |
| 02 | micrograd: a tiny autograd engine | 2026-11-15 | My backpropagation engine gives the same gradients as PyTorch | [journal](02-micrograd/journal.md) | [learnings](02-micrograd/learnings.md) | [micro_grad.ipynb](02-micrograd/micro_grad.ipynb) |
| 03 | makemore part 1: a bigram language model | 2026-11-22 | Counting model loss 2.4544; a neural network trained to about 2.46 | [journal](03-makemore-bigram/journal.md) | [learnings](03-makemore-bigram/learnings.md) | [makemore.ipynb](03-makemore-bigram/makemore.ipynb) |
| 04 | makemore part 2: an MLP language model | 2026-11-26 | Test loss about 2.15, against about 2.45 for the bigram model | [journal](04-makemore-mlp/journal.md) | [learnings](04-makemore-mlp/learnings.md) | [makemore_mlp.ipynb](04-makemore-mlp/makemore_mlp.ipynb) |

Lower loss is better. Guessing uniformly over the 27 characters gives a loss of 3.30.

## Check any project on its own

Each journal (except 01) ends with a "How to verify" section. Open the notebook, choose Restart Kernel and Run All Cells, and compare with the numbers there. Project 04 is seeded, so the same steps should reproduce its numbers on the same setup; a different PyTorch version or computer can differ slightly.

## Setup (once)

You need Python 3.10 or newer. From this folder, in a terminal:

```
./setup.sh
```

It creates `.venv`, installs `requirements.txt` (Jupyter, numpy, matplotlib, graphviz, PyTorch) and registers a kernel called "Python (ai-learning)". Then either:

- run notebooks in the browser: `source .venv/bin/activate && jupyter lab`
- or in VS Code, open this folder and pick the `.venv` interpreter as the kernel

micrograd's graph drawing also needs the Graphviz program: `brew install graphviz`. Each project folder also has its own `requirements.txt`, for running just that folder.
