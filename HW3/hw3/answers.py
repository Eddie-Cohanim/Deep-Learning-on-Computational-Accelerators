r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    pass
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    pass
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**
"""

part1_q2 = r"""
**Your answer:**
"""

part1_q3 = r"""
**Your answer:**
"""

part1_q4 = r"""
**Your answer:**
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=16,
        h_dim=1024,
        z_dim=128,
        x_sigma2=0.01,
        learn_rate=0.0001,
        betas=(0.9, 0.999),
    )
    # ========================
    return hypers


part2_q1 = r"""
**Answer:**$\\$
The $\sigma^2$ hyperparameter (x_sigma2) represents the variance of the Gaussian likelihood distribution $p_{\beta}(\mathbf{X} | \mathbf{Z})$. 
It controls the reconstruction term in the VAE loss.$\\$

When $\sigma^2$ is small, the reconstruction error is weighted much more heavily, so the model prioritizes matching the input closely. 
This can reduce the influence of the KL term, which may cause overfitting and a less regular, less smooth latent space.$\\$

When  $\sigma^2$ is large, reconstruction errors are penalized less, so the KL term has more influence. This typically produces a 
more regular latent space, but reconstructions can become blurrier or less accurate, and samples may be more diverse but lower quality.$\\$

"""

part2_q2 = r"""
**Answer 1:**$\\$
Reconstruction loss checks how close the output is to the input. Its job is to make sure the latent code keeps enough 
information so the decoder can rebuild the image well. If this term is doing its job, reconstructed images look similar
to the originals, and details that matter for the data should be preserved. Without this term, the model would have no 
strong reason to produce accurate reconstructions.$\\$

KL divergence loss is a regularizer on the latent space. It prevents the encoder from using the latent variables 
in an arbitrary way that only works for the training set thereby overfitting it. It pushes the encoder's latent distributions to stay close 
to a simple prior distribution (usually a standard normal).$\\$

**Answer 2:**$\\$
The KL term encourages the encoder to produce latent codes that look like they were sampled from the prior distribution, 
instead of being scattered unpredictably. In practice, it pulls the latent means toward zero and discourages the variances from becoming extreme.
This means the KL term creates pressure for a more consistent and shared structure across all examples, rather than every input having its own isolated 
latent region.$\\$

**Answer 3:**$\\$
It makes the mode learn a smoother, more continuous representation where small changes in the latent code lead to small, 
gradual changes in the output. It also reduces overfitting by preventing the model from memorizing the training examples, which helps it generalize 
better to new, unseen inputs.$\\$
"""

part2_q3 = r"""
**Answer:**$\\$
We start from the evidence $p(X)$ (equivalently $\log p(X)$) because we want the model to assign high probability to the observed data so 
it can later generate similar samples. However, directly optimizing $\log p(X)$ is not computationally feasible. Therefore, we maximize a
lower bound on $\log p(X)$. We aim to make this bound as tight as possible (i.e., as close to $\log p(X)$ as we can),
because a tighter bound yields a more accurate fit to the data distribution and improves the quality of the generated samples.$\\$

"""

part2_q4 = r"""
**Answer:**$\\$
We model the log-variance of the latent space rather than the variance itself mainly for stability and convenience during training: 
variance must be strictly positive, but a neural network can output any real number, so predicting variance directly would 
require extra constraints or special handling that can be numerically unstable, especially when values get very small or very large. 
By predicting log-variance instead, the network can output any real value and we convert it to a valid positive variance. This also makes the KL 
divergence term easier and more stable to compute, since the KL formula already includes a log-variance component, so having it directly avoids 
additional log operations.$\\$
"""


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0,
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    hypers = dict(
        embed_dim=128,
        num_heads=4,
        num_layers=3,
        hidden_dim=256,
        window_size=16,
        droupout=0.1,
        lr=0.0001,
    )
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**

Stacking encoder layers with sliding-window attention results in a broader context in the final layer through a mechanism similar to how stacking CNN layers increases the receptive field.

In a single layer with sliding-window attention of size $w$, each token can only attend to tokens within a distance of $w/2$ from itself. However, when we stack multiple layers:

**Layer 1**: Each token position receives information from tokens within $w/2$ distance.

**Layer 2**: Each token position now receives information from the output of Layer 1. Since Layer 1's output at each position already incorporated information from $w/2$ neighbors, Layer 2 effectively receives information from tokens up to $w$ distance away (the neighbors of neighbors).

**Layer 3**: Each token can now access information from tokens up to $3w/2$ distance away, and so on.

**Mathematically**: After $L$ layers with window size $w$, each token position can theoretically access information from tokens up to a distance of approximately $L \cdot w/2$.

This is analogous to CNNs where:
- A single convolutional layer with kernel size $k$ has a receptive field of size $k$
- Stacking $L$ such layers results in a receptive field of approximately $L \cdot k$

Therefore, by stacking multiple encoder layers, we can achieve long-range dependencies while maintaining the computational efficiency of $O(nw)$ per layer, resulting in overall complexity of $O(Lnw)$ instead of $O(n^2)$ for full attention.
"""

part3_q2 = r"""
**Your answer:**

One effective variation is **Dilated Sliding Window Attention** (inspired by dilated convolutions):

**Proposed Pattern:**
Instead of attending to consecutive tokens within a window, use a sliding window with dilation. For a token at position $i$ with window size $w$ and dilation rate $d$:
- Attend to tokens at positions: $i - d \cdot w/2, i - d \cdot (w/2-1), ..., i, ..., i + d \cdot (w/2-1), i + d \cdot w/2$

**Time Complexity:**
- Each token still attends to exactly $w$ other tokens (the window size remains fixed)
- Total complexity per layer: $O(nw)$, same as regular sliding window
- With $L$ layers: $O(Lnw)$

**Global Information Sharing:**
- **Single layer with dilation $d$**: Each token accesses information from tokens up to distance $d \cdot w/2$
- **Stacking layers with increasing dilation** (e.g., $d=1, 2, 4, 8, ...$):
  - Layer 1 ($d=1$): Access up to $w/2$ distance
  - Layer 2 ($d=2$): Access up to $2w/2 = w$ distance
  - Layer 3 ($d=4$): Access up to $4w/2 = 2w$ distance
  - Layer $k$ ($d=2^{k-1}$): Access up to $2^{k-1} \cdot w/2$ distance

This achieves **exponential growth** in receptive field with linear number of layers, requiring far fewer layers than regular sliding window to capture long-range dependencies.

**Advantages:**
- Faster global information propagation (logarithmic layers needed for sequence-length coverage)
- Same computational complexity as sliding window
- More efficient for long sequences

**Limitations:**
- May miss fine-grained local interactions that fall between dilated positions
- Requires careful tuning of dilation rates for each layer
- Information flow is still limited by the dilation pattern - some token pairs may need many layers to interact
"""

# ==============
