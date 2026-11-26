# What I learned from micrograd

Karpathy's micrograd lecture builds a tiny neural network library from scratch, in about 100 lines of Python. I coded along in my own notebook. These are the five ideas I'd want explained to me if I were starting over.

## 1. The gradient flows backward

A neural network is a long chain of small steps. Numbers go in, get multiplied and added many times, and one number comes out at the end. That last number is the loss, and it tells you how wrong the network was.

What we want to know is this: if I nudge one of the numbers inside the chain a tiny bit, how much does the loss change? That answer is the gradient.

Try it on the smallest case. Say `c = a * b` with `a = 2` and `b = 3`, so `c = 6`. Nudge `a` up to 2.1 and `c` becomes 6.3. It moved 3 times as much as the nudge, so the gradient of `a` is 3, which happens to be `b`. Nudge `b` to 3.1 instead and `c` becomes 6.2, so the gradient of `b` is 2, which is `a`.

You can't see this from the front of a big network. You work it out from the back. Start at the loss and step back one operation at a time, asking each time how much the inputs to that step mattered. The gradient travels from the end of the calculation to the beginning, and that's all "backpropagation" means.


## 2. The chain rule, and why gradients add up

The chain rule is how you step backward. If `a` changes `c` at some speed, and `c` changes `d` at some speed, then `a` changes `d` at the product of the two. If `c` moves 3 times as fast as `a`, and `d` moves 2 times as fast as `c`, then `d` moves 6 times as fast as `a`. Rates multiply as you go back.

The second half is easy to miss. When one number is used in more than one place, its effects add up.

Take `d = a * b + a`, with `a = 2` and `b = 3`. The number `a` shows up twice. Through the product it changes `d` by `b`, which is 3. Through the plain `+ a` it changes `d` by 1. The total is 4. Run it with micrograd's `Value` class and `a.grad` comes out as 4.

In the code this is the difference between `+=` and `=`. With `=`, the second use of `a` overwrites the first, and you get a wrong answer with no error to warn you.

## 3. Why we sort the nodes first

Backward only works if you go in the right order. The rule is simple: don't send a node's gradient further back until every node that uses it has reported its share.

In the example above, `a` is used by `c` and by `d`. If you handled `a` before both of them had reported, its gradient would still be incomplete. Anything feeding into `a` in a bigger network would then get the wrong number.

A topological sort puts the nodes in a list where every node comes after the ones it depends on. Here that's `a, b, c, d`. Walk that list in reverse and you get `d, c, b, a`, and the rule holds by itself, because `a` comes last, after `d` and `c` have both reported.

In micrograd this takes about ten lines. A small function visits a node's inputs first and adds the node to a list afterward.

## 4. Loss functions

A loss function squeezes "how wrong was the network" into one number. There's more than one way to do it. In the lecture it's squared error: take the prediction minus the right answer, then square it. Cross-entropy is another, and it's what language models use.

What matters is that lower means better and there's a floor, so training has somewhere to stop. Squaring gives you both. Without it, an error of +2 and an error of −2 add up to 0 and look perfect. Squared, they add up to 8. The usual losses, squared error and cross-entropy, never go below zero, and zero means a perfect answer.

## 5. Gradient descent

Once every weight has a gradient, you update them all with one line:

```
weight = weight - learning_rate * gradient
```

The gradient points uphill, toward more loss, and we want less. That's why there's a minus.

The learning rate is a small number like 0.01. The gradient only describes the slope right where you're standing, so you take small steps.

Here are real numbers. One weight, `w = 1`, input `x = 2`, and the right answer is 10. The prediction is 2, so the loss is (2 − 10)² = 64. The gradient is −32. The update gives `1 - 0.01 * (-32) = 1.32`. After 10 steps `w` is about 3.26 and the loss is about 12. After 100 steps `w` is about 5.0 and the loss is close to 0, which makes sense, since 5 × 2 = 10.

Training is that loop repeated: predict, compute the loss, go backward to get gradients, update the weights. Gradients have to be reset to zero before each backward pass, because of what we saw in section 2. They add up, and old ones would pile onto the new ones.

## Questions I can answer

Questions I'd expect in an AI engineering interview on how a neural network learns, answered in plain English.

**1. What is a gradient, and what does backpropagation compute?**
A gradient says how much the final loss changes if you nudge one number a tiny bit. Backpropagation works out that answer for every number in the network at once, by starting at the loss and stepping backward through each operation.

**2. Why does the gradient flow backward and not forward?**
The loss sits at the end of the chain, and the chain rule lets you compute each step's effect from the step after it. Starting at the end means one backward sweep gives you every gradient. Going forward, you'd have to redo the work for each number separately.

**3. What is the chain rule, in one sentence?**
If `a` changes `c` at one rate and `c` changes `d` at another, then `a` changes `d` at the two rates multiplied.

**4. Why does micrograd use `+=` for gradients and not `=`?**
A number that's used in more than one place gets gradient from every use, and those effects add up. With `=`, the last use would overwrite the others, and you'd get a wrong answer with no error.

**5. Why do you sort the nodes before the backward pass?**
A node must not pass its gradient back until every node that uses it has reported its share. A topological sort lists each node after the ones it depends on, so walking that list in reverse guarantees the order.

**6. What happens if you forget to zero the gradients between training steps?**
They accumulate. The second step's gradient would include the first step's, so the weights get pushed by stale information.

**7. What is gradient descent, and what does the learning rate do?**
After the backward pass you move each weight a small step against its gradient, which lowers the loss. The learning rate sets the step size. Too small and training crawls. Too large and you overshoot, so the loss bounces around or blows up.
