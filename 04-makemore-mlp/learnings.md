# makemore part 2 (MLP): learnings

What I learned building a small neural language model, as questions I'd expect in an AI engineering interview, answered in plain English. Most of these started as questions I asked while working. The results are in the notebook in this folder.

**1. Why does `C[X][13, 2]` equal `C[X[13, 2]]`?**
Indexing a table with a grid of numbers looks up each number on its own and keeps the grid's shape. So `C[X]` is the whole grid with every letter replaced by its row, and picking spot (13, 2) from it gives the row for the letter at that spot. Picking the letter first and then looking it up gives the same row.

**2. What is an embedding, and why use one?**
Each letter gets a short row of numbers that the model learns during training. The three rows for the three previous letters are joined and fed to the hidden layer. This keeps the input small, and letters that behave alike can end up with similar rows. The alternative is a table with one entry per possible 3-letter context: 27 x 27 x 27 contexts x 27 outputs is 531,441 numbers, and most contexts would be rare or missing. My whole model has 24,437.

**3. Why do you flatten with `emb.view(-1, n_inputs)`?**
`emb` has shape (examples, 3 letters, embedding size). The hidden layer wants one row of numbers per example, so the three letters' rows are laid out side by side: 3 x 30 = 90 numbers per example.

**4. What is a mini-batch, and why use it?**
A small random sample of the training data (32 examples) used for one step. The gradient from a sample is noisy but points about the right way, and each step costs a fraction of a full pass over 182k examples. Many cheap, rough steps get downhill faster than a few exact ones. The cost is a jittery loss that never settles exactly.

**5. Why can the loss printed at the end of training differ from the evaluation loss?**
The loss left over after the training loop is the loss on the last mini-batch only, a noisy sample of 32. The eval cells compute it on the entire set. Compare full-set numbers with full-set numbers. To watch training progress, average the last stretch of losses instead of reading one.

**6. Why split into train, dev and test?**
Loss on data the model trained on can look good because it memorized it. Held-out data shows how it handles new examples. Train updates the weights. Dev lets me compare settings (learning rate, sizes) without touching the weights. Test is run once at the end for an honest final number. If I tuned on test, it would stop being a fair test.

**7. Do I run backward on dev or test?**
No. For dev and test I only run the forward pass and compute the loss. Backward and weight updates happen on train only. After choosing settings on dev, one may retrain on train and dev together, but never on test.

**8. Train loss and dev loss are close. What does that mean?**
If both are close and both are high, the model is underfitting: it hasn't learned enough yet, and more training or capacity helps. If train is far below dev, it is overfitting, which means memorizing the training names. Mine ended with a small gap between train and dev, which is normal.

**9. What does it mean when test loss is about equal to dev loss?**
I picked settings by watching dev, which can quietly fit dev. Test is data I never tuned on, so a matching number is evidence that the dev result is honest and is a fair estimate for new names.

**10. How does hidden-layer size matter?**
It sets how many patterns the model can detect between reading the letters and guessing the next one. Too few neurons and both train and dev stay high however long I train. Too many and it can memorize, which shows up as train falling while dev stalls or rises. It also sets the parameter count: the weights into and out of the hidden layer both scale with it.

**11. How do you pick a learning rate?**
Sweep: try rates from small to large (for example 10 to the power of -3 up to 1), take one step per rate, and plot loss against rate. Pick near the bottom of the valley, before the loss blows up. Then drop the rate later in training so the weights settle. I trained in six passes of 30,000 steps: two at 0.1, two at 0.01, two at 0.005. Too high and the loss bounces or explodes. Too low and training crawls.

**12. Why did I get `element 0 of tensors does not require grad`?**
The parameters weren't tracking gradients. Setting `requires_grad = True` on each parameter lets PyTorch record the steps and compute `.grad` in `loss.backward()`. Without it the loss has no history to go backward through.

**13. What was wrong with `p.data = -0.1 * p.grad`?**
It replaces each weight with its scaled gradient, wiping out everything learned. A gradient descent step moves the existing weight a small way against its gradient: `p.data += -lr * p.grad`.

**14. Why compare against the bigram model and the uniform baseline?**
They give the number meaning. Uniform guessing over 27 letters gives a loss of ln(27), about 3.30. The counting bigram model is the next step up. The MLP's test loss is clearly lower than both. Without baselines, a loss number on its own tells you nothing.

## Questions I can't answer yet

- How much of the improvement comes from the embedding itself, as opposed to the longer context and the hidden layer? I didn't run the model with and without it.
- What each of embedding size, hidden size and batch size contributed on its own. I changed them together once.
- Why unscaled random starting weights give a high first loss and what better starting values look like (a later lecture).
- How to choose a batch size by principle rather than by habit.
