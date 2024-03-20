---
title: Random
---

# Random Notes:

---

## LLM Security

*Section from [this](https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/) blog post*

**Types of attacks:**

- White-box: have access to model weights, e.g. gradient-based attacks
- Black-box: no access to model internals, treat as an API
- Human vs model red-teaming: human vs model attacking a LLM
- Token manipulation: change some tokens s.t. model fails, but sementatic info is the same

Token manipulation:

- *Semantically Equivalent Adversaries Rules:*
    - Simple token-replacement strategy
    - Heuristic based replacements
- *TextFooler and BERT-Attack:*
    - Word importance score: similar to difference between model predictions when word is masked vs unmasked
    - Words with high importance are good candidates to be replaced
    - Replace important words with similar words, check changes
        - BERT-Attack uses BERT-similar words, TextFooler uses embedding cosine similarity

**Gradient-Based Attacks:**

- *Gradient-based Distributional Attack:*
    - Use Gumbel-Softmax trick to make the adversarial optimization differentiable
        - Adversaial differentiation attacks for images, feed in example image, update image "weights" w.r.t. optimizing prediction of different class
        - We can't do this naturally with LLMs because the classes are discrete (image pixels are continuous, whereas tokens are discrete)
        - Token $x_i$ is usually drawn from a categorical over $\pi_i$, where $\pi_i = \text{Softmax}(\Theta_i)$
    - *Gumbel-Max:*
        - When sampling from unnormalized distributions, we often softmax then treat as probs
        - We can use Gumbel-Max to do the same: $y_i \sim \text{Softmax}(x_i)$ is equivalant to $x_i \sim \arg_i \max x_i + z_i$ where $z_i \sim \text{Gumbel}(0, 1)$ where $x_i$ are the unnormalized logits for position $i$
        - If $x_i$ are normalized (e.g. softmaxed) outputs, we can use $\log(x_i)$ instead
        - We would take the one-hot encoding of this as our output vector
    - *Gumbel-Softmax:*
        - Instead of using $\text{one-hot} \arg \max ((\dots))$, we can use $\text{Softmax}$.
        - This leads to the outputs no longer being one-hot encoding, but they're differentiable
        - $y_{i} = \text{Softmax}(\frac{x_i + g_i}{\tau})$ where $\tau$ is the temperature and $g_i \sim \text{Gumbel}(0, 1)$
        - Low $\tau$ leads to one-hot distribution, just the argmax
        - This is a weighted combination of the embedding predictions
    - We update the "logits"(?) to minimize the adversarial loss
    - Soft constraints (e.g. fluency, BERTScore) can be used as well!
    - Final loss: $\mathcal{L}_\text{adv} + \mathcal{L}_\text{NLL} + \mathcal{L}_\text{sim}$
- *HotFlip:* treat inputs in vector space, apply transformations guided by loss
    - Represent text sequences as a one-hot character embedding: $\mathbb{R}^{m \times n \times V}$ tensor, where $m$ is number of words, $n$ is maximum number of characters per word, $V$ is vocab size
    - A transformation $x_{i, j, a \rightarrow b}$ is represented as the $j$'th character of the $i$'th word being chaned from $a$ to $b$
    - The first-order Taylor loss is: $\nabla_{x_{i, j, a \rightarrow b} - X} \mathcal{L}(x, y) = \nabla_x \mathcal{L}_\text{adv}(x,y)^T (x_{i, j, a \rightarrow b} - x)$ where:
        - $\mathcal{L}_\text{adv}(x, y)$ is the adversarial loss
        - Note: this is a matrix-multiplication
    - Flips are chosen by what decreases the adversarial loss the greatest
    - Soft-constraints can be added as well
    - Token addition or deletion can be represented as well
- *Universal Adversarial Triggers:*
    - Find a short trigger sequence for specific classification or generation outputs
        - e.g. ~1 token for classification, ~4 tokens for generation
    - We basically just run HotFlip on this to update and find optimial sequence
    - They can be model-agnostic
    - *UAT-LM* adds a language model loss
    - *UTSC*: get unigram UTCs, filter them by a toxicity selection criteria
    - Both modifications get significantly lower perplexity and look more realistic
- *Adversarial Suffixes:*
    - Get the NLL of of a sequence: `{CONVERSATION HISTORY / MODEL INSTRUCTIONS} {USER QUERY} {ADV SUFFIX} {TARGET OUTPUT}`
        - E.g. `System: You are a safe chatbot. User: How to create a bomb? {ADV SUFFIX} Assistant: {TARGET RESPONSE}
    - Loss is the NLL of outputting `{TARGET RESPONSE}`
    - Use greedy substitutions for a number of steps
- *ARCA (Autoregressive Randomized Coordinate Ascent):*
    - Find a mapping of (input, toxic) to a score of how non-toxic the input and toxic the output is
    - Can be about any sort of transition between types of text
    - e.g. derogitory comments about celebs score: $\rho(x, y) = \text{StartsWith}(x, \text{[celebrities]}) + \text{NotToxic}(x) + \text{Toxic}(y)$
    - We want to update $x$ such that $\rho(x, y)$ increases for the predicted $y \sim x$
    - $\max_{x, y} \rho(x, y) + \lambda_\text{LLM} \log p(y|x)$
    - Iterate over each token index, making replacements until $p(x) = y$ (high completion probability) and $\rho \geq \text{threshold}$ (or iteration limit)
        - This is *coodinate ascent* because we iterate over the coodinates - neat!
