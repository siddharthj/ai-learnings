# AI Learning Journal

Notes and notebooks from learning how large language models are built from scratch, following Andrej Karpathy's [Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero) series. Each project has its own folder with the notebook, requirements, journal and learnings, so it can be checked on its own.

Credit: the lectures, the micrograd engine and the `names.txt` data set are Andrej Karpathy's ([nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero), [micrograd](https://github.com/karpathy/micrograd), [makemore](https://github.com/karpathy/makemore)). The notebooks are my own code written while following the lectures. The journals and learnings are my notes.

## Projects

| # | Project | Date | Journal | Learnings | Notebook |
|---|---|---|---|---|---|
| 01 | Karpathy: Intro to LLMs and Deep Dive into LLMs (watched, no code) | 2026-11-11 | [journal](01-intro-videos/journal.md) | n/a | n/a |
| 02 | micrograd: a tiny autograd engine | 2026-11-15 | [journal](02-micrograd/journal.md) | [learnings](02-micrograd/learnings.md) | [micro_grad.ipynb](02-micrograd/micro_grad.ipynb) |
| 03 | makemore part 1: a bigram language model | 2026-11-22 | [journal](03-makemore-bigram/journal.md) | [learnings](03-makemore-bigram/learnings.md) | [makemore.ipynb](03-makemore-bigram/makemore.ipynb) |
| 04 | makemore part 2: an MLP language model | 2026-11-26 | [journal](04-makemore-mlp/journal.md) | [learnings](04-makemore-mlp/learnings.md) | [makemore_mlp.ipynb](04-makemore-mlp/makemore_mlp.ipynb) |

## Check any project on its own

Each journal ends with a "How to verify" section: how to run the notebook and the numbers a clean run should produce. Open the notebook, choose Restart Kernel and Run All Cells, and compare.

## Setup (once)

From this folder, in a terminal:

```
./setup.sh
```

It creates `.venv`, installs `requirements.txt` (Jupyter, numpy, matplotlib, graphviz, PyTorch) and registers a kernel called "Python (ai-learning)". Then either:

- run notebooks in the browser: `source .venv/bin/activate && jupyter lab`
- or in VS Code, open this folder and pick the `.venv` interpreter as the kernel

micrograd's graph drawing also needs the Graphviz program: `brew install graphviz`. Each project folder also has its own `requirements.txt`, for running just that folder. A new library for a later notebook goes in the root `requirements.txt` and the project's own, then re-run `./setup.sh`.

## Adding a new project

1. Make a folder `NN-short-name/`.
2. Copy `_template/journal.md` and `_template/learnings.md` into it and fill them in while the session is fresh. Add a "How to verify" section to the journal.
3. Put the notebook, any data, and a `requirements.txt` next to them.
4. Add a row to the table above.

## Other files

- `_template/` - the journal template for new projects
