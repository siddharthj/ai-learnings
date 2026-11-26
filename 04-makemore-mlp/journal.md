# 2026-11-26: Karpathy, makemore part 2 (MLP)

**Source:** Andrej Karpathy, makemore part 2 (Zero to Hero, lecture 3), from the [nn-zero-to-hero page](https://github.com/karpathy/nn-zero-to-hero)
**Dates:** started 2026-11-23 (folder created), finished 2026-11-26 (test set run)
**Notebook / code:** [`makemore_mlp.ipynb`](makemore_mlp.ipynb), data in [`names.txt`](names.txt) (from [Karpathy's makemore repo](https://github.com/karpathy/makemore))

## What I did and the result
Built a character-level language model that predicts the next letter from the previous three, using an embedding table and one hidden layer. Trained it with mini-batch gradient descent, split the names into train, dev and test, and tuned on dev.

| What | Value |
|---|---|
| Names in the data set | 32,033 |
| Examples (train / dev / test) | 182,455 / 22,929 / 22,762 |
| Context length | 3 letters |
| Embedding size, hidden neurons, batch size | 30, 200, 32 |
| Training | 6 passes of 30,000 steps (180,000 total): 2 at learning rate 0.1, 2 at 0.01, 2 at 0.005 |
| Parameters | 24,437 (27x30 + 90x200 + 200 + 200x27 + 27) |
| Uniform-guessing baseline loss | ln(27) = 3.30 |
| Counting bigram model (last project) | 2.4544 |
| **Train loss (full set)** | **2.0789** |
| **Dev loss (full set)** | **2.1693** |
| **Test loss (run once, at the end)** | **2.1709** |

The MLP beats the bigram by about 0.28 on unseen names, because it sees three letters of context instead of one. Test (2.1709) is almost the same as dev (2.1693). I tuned on dev but never on test, so about 2.17 is an honest estimate for new names.

Sample names from the trained model: `deliah.`, `kaleigh.`, `tate.`, `skanden.`, `quintin.` Some are not names at all (`kentceriin.`, `dearynix.`).


## How I did it
Coded along with the lecture in my own notebook. Wrote the dataset builder (3 letters in, next letter out), the embedding table `C`, a hidden layer with `tanh`, and an output layer with cross-entropy. Trained on random mini-batches of 32. Moved every setting into one cell at the top. Tried a learning-rate sweep to find a good rate. Trained in six passes of 30,000 steps each, every pass continuing from the weights the last one left: two at learning rate 0.1, two at 0.01, two at 0.005. That schedule produced the numbers above.

## What confused me, and how I got unstuck
- **`C[X][13,2]` equals `C[X[13,2]]`.** Indexing is applied to each number separately. Looking up the table for the whole grid and then picking one spot gives the same row as picking the spot first and looking that one up.
- **`RuntimeError: element 0 of tensors does not require grad`, again.** Same cause as makemore part 1: the parameters need `requires_grad=True`, or the loss has no history to go backward through. Fixed by setting it on every parameter right after creating them.
- **Weights that went wrong.** The update line was `p.data = -0.1 * p.grad`, which replaced each weight with its scaled gradient. It needs `+=`, so each weight moves a small step from where it already is. Also had a typo in a variable name.
- **The loss at the end of the training loop is not the model's loss.** It is the loss on the last mini-batch of 32 examples, a noisy sample. The eval cells compute the loss over the whole set, so those are the numbers to compare. Re-run the eval cells after every training pass, or they show an older model.
- **Re-running the split cell gives a different split.** That cell shuffles the name list in place, so running it again shuffles an already shuffled list. Restart and run from the top for a repeatable split.
- **The settings cell and the training loop disagreed.** The loop had a hard-coded learning rate, so changing the settings cell did nothing. I made the loop read `learning_rate`.

## What I learned
- An embedding gives each letter a short list of numbers that the model learns. It keeps the input small and lets similar letters end up with similar numbers. A table for every 3-letter context would need 27 x 27 x 27 x 27 = 531,441 numbers, against 24,437 here.
- Mini-batches trade a noisy gradient for many cheap steps. The loss printed at the end of a loop is one noisy batch, so I compare full-set losses only.
- Train, dev and test do three different jobs. Train updates the weights. Dev picks settings. Test is read once at the end.
- Train close to dev, both high, means the model is still underfitting. A big gap, with train much lower, means it is memorizing.
- Hidden size sets capacity. Too small and both losses stay high. Too large and it can memorize. Learning rate and training length mattered more for my first runs than hidden size did.
- Change one setting at a time. I changed embedding size, hidden size and batch size together once, and then couldn't tell which one had caused the change.

**Quiz:** 8 interview-style questions on this project. Questions and answers are in [`learnings.md`](learnings.md).

## Limits and what I'd do differently
- Only three letters of context, and some samples are still not names (`kentceriin.`, `dearynix.`).
- The weights start as plain unscaled random numbers (`torch.randn`). I did not tune the starting values.
- I picked the learning-rate schedule by hand while watching dev loss, and wrote it down only at the end. Next time I will log each pass as I go.
- One run, one random seed. I don't know how much the numbers move between seeds.
- I did not test the embedding on its own (same model with and without it), so I can't say how much of the gain comes from the embedding and how much from the extra context and the hidden layer.
- I tried a few sizes by hand, judged by dev loss alone. It was not a proper search.

## What's next
Next lecture in the series.

## How to verify
Install `requirements.txt`, open `makemore_mlp.ipynb` from this folder so `names.txt` is found, and run the cells from the top. To follow my schedule, set `learning_rate` in the settings cell, run the training cell twice at 0.1, twice at 0.01 and twice at 0.005 (each run continues from the weights the last one left), then run the three eval cells.

My saved results are train 2.0789, dev 2.1693 and test 2.1709, with 182,455 / 22,929 / 22,762 examples. Because the split cell reshuffles in place, a run from the top uses a different split, so expect the losses to differ by a few hundredths.

What a correct run always shows:
- The loss falls over the training passes.
- Train and dev end within about 0.1 of each other.
- Test is close to dev, because test was never used for tuning.
- Comparisons use full-set losses, not the last mini-batch.
