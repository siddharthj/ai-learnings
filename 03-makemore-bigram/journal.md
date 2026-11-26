# 2026-11-22: Karpathy, makemore part 1 (bigram language model)

**Source:** Andrej Karpathy, makemore part 1 (Zero to Hero, lecture 2), from the [nn-zero-to-hero page](https://github.com/karpathy/nn-zero-to-hero)
**Dates:** started 2026-11-21, finished 2026-11-22 (from my notebook's file dates)
**Notebook / code:** [`makemore.ipynb`](makemore.ipynb), data in [`names.txt`](names.txt) (from [Karpathy's makemore repo](https://github.com/karpathy/makemore))

## What I did and the result
Built a character-level language model that generates new names, two ways, then compared them.

1. **Counting model.** Counted how often each letter follows another over 32,033 names, turned the counts into probabilities, and sampled names from them.
2. **Neural network.** A single weight matrix `W` (27 x 27 = 729 numbers), trained with gradient descent on 228,146 letter pairs to predict the next letter.

| What | Value |
|---|---|
| Names in the data set | 32,033 |
| Training examples (letter pairs) | 228,146 |
| Uniform-guessing baseline loss | ln(27) = 3.30 |
| Counting model loss (with +1 smoothing) | 2.4544 |
| Trained network loss | about 2.46 (200 steps, learning rate 50) |
| Parameters in the network | 729 |

Both models generate plausible-looking but nonsense names, because each prediction sees only one letter of context.

## How I did it
Coded along in my own notebook. Computed the average negative log likelihood to score each model, then trained the network with the same loop as micrograd: forward pass, loss, backward pass, update.

**How I checked it:** the counting model and the neural network should land on nearly the same loss, since both can represent the same table of probabilities. They do (2.4544 and about 2.46). A clean top-to-bottom run of the notebook reproduces the numbers.

## What confused me, and how I got unstuck
- **`RuntimeError: element 0 of tensors does not require grad`.** I had created `W` without `requires_grad=True`, so PyTorch recorded nothing to differentiate. Fixed by creating it with `requires_grad=True`, then rerunning the forward pass before `backward()`.
- **`Trying to backward through the graph a second time`.** I ran `loss.backward()` twice on the same loss. `backward()` frees the recorded graph, so each backward pass needs a fresh forward pass first.
- **Wrong first letter in the generated names.** A leftover `torch.multinomial` line before the loop drew one random number and threw it away, so every later draw shifted by one. Removing it gave the lecture's names (`cexze.` and so on).
- **What `torch.multinomial` does.** It rolls a weighted die over a list of probabilities and returns the index it lands on. I use it instead of picking the biggest probability so the model gives different names each time.
- **Where `W.grad` is calculated.** Inside `loss.backward()`, by PyTorch, using the same chain rule I wrote by hand in micrograd. The gradient only exists after the forward pass has produced a loss, because it measures how that loss changes.

## What I learned
- A language model is a table of "what comes next" probabilities. Counting builds the table directly. A neural network learns nearly the same table by gradient descent.
- Loss is the average negative log probability of the correct next letter. Logs avoid multiplying thousands of tiny numbers, and the baseline for guessing uniformly over 27 characters is ln(27), about 3.30.
- Smoothing (adding 1 to every count) stops a never-seen letter pair from giving probability 0, which would make the loss infinite. Penalizing the size of `W` does the same job in the network.
- `one-hot @ W` just picks one row of `W`, and softmax (`exp`, then divide by the row sum) turns that row into probabilities.
- Sampling from the probabilities gives variety. Taking the most likely letter every time would repeat the same name.
- Checking one implementation against another (counting vs. network) is a good way to catch bugs.

**Quiz:** 9 interview-style questions on backprop and this model. Questions and answers are in [`learnings.md`](learnings.md).

## Limits and what I'd do differently
- A bigram model sees one letter back, so it can't learn word shape beyond pairs. The next lectures widen the context.
- I didn't add regularization, so the network has no equivalent of smoothing. The lecture suggests it as an exercise.
- 200 steps at learning rate 50 was taken from the lecture. I didn't test other settings.

## What's next
Next lecture in the series. Before the tokenizer, I plan to finish the rest of makemore and "Let's build GPT".

## How to verify
Install `requirements.txt` and open `makemore.ipynb` from this folder, so `names.txt` is found. Choose Restart Kernel and Run All Cells. The numbers are in the table at the top of this entry. Also expect the counting model's first sampled names, with seed 2147483647, to be `cexze.`, `momasurailezitynn.`, `konimittain.`, `llayn.`, `ka.`

The network's samples can differ slightly from the counting model's, depending on how long it trained. Short comments were added and no code was changed.
