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
    hypers['batch_size'] = 64
    hypers['seq_len'] = 80
    hypers['h_dim'] = 100
    hypers['n_layers'] = 3  
    hypers['dropout'] = 0.2
    hypers['learn_rate'] = 0.002
    hypers['lr_sched_factor'] = 0.1
    hypers['lr_sched_patience'] = 4
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "ACT I."
    temperature = 0.5
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Answer 1:**
Very long sequences make training computationally expensive and unstable while using shorter sequences keeps memory usage 
manageable, allows efficient batching and parallel training, and enables truncated backpropagation through time, which stabilizes 
gradients. In addition, splitting the text creates many training samples from a single corpus, improving data efficiency
and generalization




"""

part1_q2 = r"""
**Answer 2:**

Even though the model is trained on fixed length sequences, it can remember information for longer than that and this is because 
the hidden state is passed forward instead of being reset. During training, batches are processed in order so the hidden state 
from one batch becomes the context for the next. During generation, the hidden state keeps accumulating information, 
allowing the model to maintain long-range context beyond the sequence length.
"""

part1_q3 = r"""
**Answer 3:**
As said before while training the network, we pass the hidden state between batches. This hidden state acts as context that summarizes the 
preceding samples. Therefore, batches must preserve the original order of the text so that consecutive batches correspond to 
adjacent parts of the corpus. If batches were shuffled, the hidden state would no longer match the input sequence and would 
effectively represent random context.
"""

part1_q4 = r"""
**Answer 4:**
The temperature controls how sharp or flat the sampling distribution is. A higher temperature produces a more uniform distribution, while a lower 
temperature produces a sharper and more spiky. Since we usually want the model to prefer the most likely next characters rather than sample almost uniformly, 
we often lower the temperature to increase confidence.

When the temperature is very high, dividing the logits by a large $T$ reduces the relative differences between them. After applying softmax, this results in a 
nearly uniform distribution, causing the model to sample characters almost at random and often generate incoherent text.

When the temperature is very low, dividing by a small $T$ amplifies the differences between logits. Because softmax is exponential, even small score 
differences become even more emplified, concentrating most of the probability mass on the highest scoring characters. This leads to more deterministic and confident 
predictions, but can also cause repetitive text.
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
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        hidden_dim=1024,
        window_size=64,
        droupout=0.1,
        lr=0.0001,
    )
    # ========================
    return hypers


part3_q1 = r"""
**Answer1:**

Stacking encoder layers with sliding-window attention increases the effective context because information can propagate across 
layers, even though each individual layer only attends locally. This is directly analogous to CNNs, where stacking convolutional 
layers increases the receptive field despite each layer having a small kernel.

In the first encoder layer, each token only attends to tokens within a fixed local window. In the next layers, tokens attend to 
representations that already include information from their neighbors windows in the previous layer. As layers are stacked, 
this local information is repeatedly combined and passed forward, allowing each token in higher layers to indirectly 
incorporate information from progressively more distant tokens.

As result, the final layer can have to a much broader context, even though every attention operation is restricted to a 
local sliding window."""

part3_q2 = r"""
**Answer 2:**

One variation is a \textbf{multi-scale sparse window}. The goal is still like sliding-window attention (each token attends to
only $w$ others), but instead of using all $w$ links on the closest neighbors, we use some of them on farther tokens so the
token gets a more global view.

So for a token at position $i$, instead of attending to the contiguous block around it, we choose neighbors at increasing
distances. For example:
\[
\mathcal{N}(i)=\{i\pm 1,\ i\pm 2,\ i\pm 4,\ i\pm 8\}\cap[1,n].
\]
This keeps the very local information (the $\pm1,\pm2$ offsets), but also adds a few longer jumps (like $\pm8$), meaning each
token can see further without attending to everything.

The complexity does not change asymptotically: each token attends to $|\mathcal{N}(i)|=O(w)$ keys, so one layer costs
\[
T_{\text{layer}} = O(nw)
\]

global information get shared through multiple layers. In one layer, token $i$ can pick up
information from a farthr token like $i+8$. In the next layer, that information is already inside the representation of
$i$, so when $i$ attends again (to $i+8$ and also to other far tokens), it can propagate that information further. In this
sense, long-range context is built by chaining hops across layers, not by a single direct attention edge to every position.

There will be fewer layers and that is because we also include larger jumps, the
distance that information can travel grows much faster. with offsets like $1,2,4,8,\ldots$ the reachable range can
grow quickly with depth, and a an estimate is that reaching the whole sequence takes about
\[
L = O(\log(n/w))
\]
layers.

our limitations are that even with the long jumps, two very far tokens can not attend to each other directly in one layer, so
their interaction is still indirect and goes through intermediate tokens. Also, since the pattern is sparse, it may
skip some positions gaps unless we mix different jump sizes across heads or rotate the pattern across layers.

"""

# ==============
