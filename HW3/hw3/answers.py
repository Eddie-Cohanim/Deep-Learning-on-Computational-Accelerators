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
    hypers['seq_len'] = 100
    hypers['h_dim'] = 128
    hypers['n_layers'] = 2  
    hypers['dropout'] = 0.2
    hypers['learn_rate'] = 0.001
    hypers['lr_sched_factor'] = 0.5
    hypers['lr_sched_patience'] = 2

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
**Answer 1:** $\\$

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
    pass
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**
"""

part2_q2 = r"""
**Your answer:**
"""

part2_q3 = r"""
**Your answer:**
"""

part2_q4 = r"""
**Your answer:**
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
    pass
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**

Stacking encoder layers with sliding-window attention increases the effective context because information can propagate across 
layers, even though each individual layer only attends locally. This is directly analogous to CNNs, where stacking convolutional 
layers increases the receptive field despite each layer having a small kernel.

In the first encoder layer, each token only attends to tokens within a fixed local window. In the next layers, tokens attend to 
representations that already include information from their neighbors windows in the previous layer. As layers are stacked, 
this local information is repeatedly combined and passed forward, allowing each token in higher layers to indirectly 
incorporate information from progressively more distant tokens.

As result, the final layer can have to a much broader context, even though every attention operation is restricted to a 
local sliding window.

"""

part3_q2 = r"""
**Your answer:**
"""

# ==============
