# makemore part 1: learnings

What I learned building a bigram language model, as questions I'd expect in an AI engineering interview, answered in plain English. The numbers come from the notebook in this folder: 32,033 names, 228,146 letter pairs.

**1. What is a bigram model, and what can't it do?**
It predicts the next character from only the one before it. It has no memory beyond that, so it can produce names that look plausible one pair at a time and still turn into nonsense overall, like `momasurailezitynn`.

**2. How do you measure how good the model is?**
Take the probability the model gave to each correct next letter, take the log of it, and average the negative of those. Lower is better. The counting model scored 2.4544 on my data.

**3. Why use logs and not multiply the probabilities?**
Multiplying thousands of numbers smaller than 1 ends up so close to zero that the computer can't tell it apart from zero. Taking logs turns the product into a sum, which stays well behaved. The minus sign turns "make this big" into "make this small," which is what training minimizes.

**4. What loss would a model get by guessing uniformly, and why does it matter?**
With 27 characters each has probability 1/27, so the loss is ln(27), about 3.30. It's a sanity check. A freshly initialised network should start near that number, a little above it. If it starts far away, something is wrong. A trained model should land well below it, and mine reached about 2.46.

**5. Why add 1 to every count (smoothing)?**
Some letter pairs never appear in the data. They'd get probability 0, and the log of 0 is infinite, so one unseen pair in a new name would give an infinite loss. Adding 1 gives every pair a small chance.

**6. What does softmax do, and why use `exp`?**
The network outputs raw scores, called logits, that can be negative or any size. `exp` makes every score positive while keeping their order, and dividing by the row total makes them add up to 1. The result is a proper probability distribution.

**7. What does one-hot encoding followed by `xenc @ W` actually do?**
A one-hot row has a single 1, so multiplying it by `W` just picks out one row of `W`. That row is the model's scores for what comes next. The weights `W` act as a lookup table with 27 x 27 = 729 numbers, the same size as the counting table.

**8. The counting model and the neural network ended up with almost the same loss. Why?**
They can represent the same thing, a table of next-letter probabilities. Training the network finds numbers that give nearly the same table, so the losses are close (2.45 for counting, about 2.46 after 200 training steps). It's also a good debugging check: if the network's loss were far from the counting loss, I'd suspect a bug.

**9. How is smoothing related to regularization?**
Adding 1 to the counts pulls every probability toward uniform. In the neural network, adding a small penalty on the size of `W` does the same job, because it also pulls the scores toward equal.

**10. Why sample from the probabilities and not always take the most likely letter?**
Taking the most likely letter gives the same name every time. Sampling with `torch.multinomial` gives varied names, and likely letters still come up most often. Language models like ChatGPT do the same thing at every step.
