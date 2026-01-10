import re
import torch
import torch.nn as nn
import torch.utils.data
from torch import Tensor
from typing import Iterator

# Yuval
def char_maps(text: str):
    """
    Create mapping from the unique chars in a text to integers and
    vice-versa.
    :param text: Some text.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the text.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """
    # TODO:
    #  Create two maps as described in the docstring above.
    #  It's best if you also sort the chars before assigning indices, so that
    #  they're in lexical order.
    # ====== YOUR CODE: ======
    unique_chars = sorted(set(text))
    char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
    idx_to_char = {idx: char for idx, char in enumerate(unique_chars)}
    # ========================
    return char_to_idx, idx_to_char


def remove_chars(text: str, chars_to_remove):
    """
    Removes all occurrences of the given chars from a text sequence.
    :param text: The text sequence.
    :param chars_to_remove: A list of characters that should be removed.
    :return:
        - text_clean: the text after removing the chars.
        - n_removed: Number of chars removed.
    """
    # TODO: Implement according to the docstring.
    # ====== YOUR CODE: ======
    chars_to_remove_set = set(chars_to_remove)
    text_clean = ''.join([char for char in text if char not in chars_to_remove_set])
    n_removed = len(text) - len(text_clean)
    # ========================
    return text_clean, n_removed


def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tensor
    corresponding to the index of that char.
    :param text: The text to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where N is the length of the sequence
    and D is the number of unique chars in the sequence. The dtype of the
    returned tensor will be torch.int8.
    """
    # TODO: Implement the embedding.
    # ====== YOUR CODE: ======
    num_chars = len(text)
    vocab_size = len(char_to_idx)
    result = torch.zeros(num_chars, vocab_size, dtype=torch.int8)

    for i, char in enumerate(text):
        char_idx = char_to_idx[char]
        result[i, char_idx] = 1
    # ========================
    return result


def onehot_to_chars(embedded_text: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_text: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """
    # TODO: Implement the reverse-embedding.
    # ====== YOUR CODE: ======
    char_indices = torch.argmax(embedded_text, dim=1)
    result = ''.join([idx_to_char[idx.item()] for idx in char_indices])
    # ========================
    return result


def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int, device="cpu"):
    """
    Splits a char sequence into smaller sequences of labelled samples.
    A sample here is a sequence of seq_len embedded chars.
    Each sample has a corresponding label, which is also a sequence of
    seq_len chars represented as indices. The label is constructed such that
    the label of each char is the next char in the original sequence.
    :param text: The char sequence to split.
    :param char_to_idx: The mapping to create and embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
    # TODO:
    #  Implement the labelled samples creation.
    #  1. Embed the given text.
    #  2. Create the samples tensor by splitting to groups of seq_len.
    #     Notice that the last char has no label, so don't use it.
    #  3. Create the labels tensor in a similar way and convert to indices.
    #  Note that no explicit loops are required to implement this function.
    # ====== YOUR CODE: ======
    # Embed the text
    embedded_text = chars_to_onehot(text, char_to_idx).to(device)

    # Calculate number of complete samples
    num_samples = (len(text) - 1) // seq_len

    # Create samples from embedded text (exclude last char since it has no label)
    samples_flat = embedded_text[:num_samples * seq_len]
    samples = samples_flat.view(num_samples, seq_len, -1)

    # Create labels (shifted by 1 position)
    labels_flat = embedded_text[1:num_samples * seq_len + 1]
    labels = torch.argmax(labels_flat, dim=1).view(num_samples, seq_len)
    # ========================
    return samples, labels


def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    # TODO: Implement based on the above.
    # ====== YOUR CODE: ======
    scaled = y / temperature
    # Subtract max for numerical stability
    scaled_max = torch.max(scaled, dim=dim, keepdim=True).values
    scaled = scaled - scaled_max
    exp_scaled = torch.exp(scaled)
    result = exp_scaled / torch.sum(exp_scaled, dim=dim, keepdim=True)
    # ========================
    return result


def generate_from_model(model, start_sequence, n_chars, char_maps, T):
    """
    Generates a sequence of chars based on a given model and a start sequence.
    :param model: An RNN model. forward should accept (x,h0) and return (y,
    h_s) where x is an embedded input sequence, h0 is an initial hidden state,
    y is an embedded output sequence and h_s is the final hidden state.
    :param start_sequence: The initial sequence to feed the model.
    :param n_chars: The total number of chars to generate (including the
    initial sequence).
    :param char_maps: A tuple as returned by char_maps(text).
    :param T: Temperature for sampling with softmax-based distribution.
    :return: A string starting with the start_sequence and continuing for
    with chars predicted by the model, with a total length of n_chars.
    """
    assert len(start_sequence) < n_chars
    device = next(model.parameters()).device
    char_to_idx, idx_to_char = char_maps
    out_text = start_sequence

    # TODO:
    #  Implement char-by-char text generation.
    #  1. Feed the start_sequence into the model.
    #  2. Sample a new char from the output distribution of the last output
    #     char. Convert output to probabilities first.
    #     See torch.multinomial() for the sampling part.
    #  3. Feed the new char into the model.
    #  4. Rinse and Repeat.
    #  Note that tracking tensor operations for gradient calculation is not
    #  necessary for this. Best to disable tracking for speed.
    #  See torch.no_grad().
    # ====== YOUR CODE: ======
    with torch.no_grad():
        hidden_state = None

        # Process start sequence
        for char in start_sequence:
            # chars_to_onehot returns (1, V), we need (B, S, V) = (1, 1, V)
            char_input = chars_to_onehot(char, char_to_idx).unsqueeze(0).to(dtype=torch.float, device=device)
            output, hidden_state = model(char_input, hidden_state)

        # Generate remaining characters
        while len(out_text) < n_chars:
            # Get probabilities from last output
            scores = output[0, -1, :]
            probabilities = hot_softmax(scores, dim=0, temperature=T)

            # Sample next character
            sampled_idx = torch.multinomial(probabilities, 1).item()
            next_char = idx_to_char[sampled_idx]
            out_text += next_char

            # Feed sampled character back to model
            char_input = chars_to_onehot(next_char, char_to_idx).unsqueeze(0).to(dtype=torch.float, device=device)
            output, hidden_state = model(char_input, hidden_state)
    # ========================

    return out_text


class SequenceBatchSampler(torch.utils.data.Sampler):
    """
    Samples indices from a dataset containing consecutive sequences.
    This sample ensures that samples in the same index of adjacent
    batches are also adjacent in the dataset.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size):
        """
        :param dataset: The dataset for which to create indices.
        :param batch_size: Number of indices in each batch.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[int]:
        # TODO:
        #  Return an iterator of indices, i.e. numbers in range(len(dataset)).
        #  dataset and represents one  batch.
        #  The indices must be generated in a way that ensures
        #  that when a batch of size self.batch_size of indices is taken, samples in
        #  the same index of adjacent batches are also adjacent in the dataset.
        #  In the case when the last batch can't have batch_size samples,
        #  you can drop it.
        idx = None  # idx should be a 1-d list of indices.
        # ====== YOUR CODE: ======
        num_batches = len(self.dataset) // self.batch_size
        idx = []

        for batch_idx in range(num_batches):
            for sample_idx in range(self.batch_size):
                dataset_idx = batch_idx + sample_idx * num_batches
                idx.append(dataset_idx)
        # ========================
        return iter(idx)

    def __len__(self):
        return len(self.dataset)


class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """

    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        """
        :param in_dim: Number of input dimensions (at each timestep).
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of input dimensions (at each timestep).
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.layer_params = []

        # ====== YOUR CODE: ======
        for layer_idx in range(n_layers):
            # Determine input dimension for this layer
            layer_in_dim = in_dim if layer_idx == 0 else h_dim

            # Create GRU parameters for this layer
            # Update gate parameters
            W_xz = nn.Linear(layer_in_dim, h_dim, bias=False)
            W_hz = nn.Linear(h_dim, h_dim, bias=True)

            # Reset gate parameters
            W_xr = nn.Linear(layer_in_dim, h_dim, bias=False)
            W_hr = nn.Linear(h_dim, h_dim, bias=True)

            # Candidate hidden state parameters
            W_xg = nn.Linear(layer_in_dim, h_dim, bias=False)
            W_hg = nn.Linear(h_dim, h_dim, bias=True)

            # Dropout layer (applied between layers, not after last layer)
            dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

            # Register parameters as modules
            self.add_module(f'layer{layer_idx}_W_xz', W_xz)
            self.add_module(f'layer{layer_idx}_W_hz', W_hz)
            self.add_module(f'layer{layer_idx}_W_xr', W_xr)
            self.add_module(f'layer{layer_idx}_W_hr', W_hr)
            self.add_module(f'layer{layer_idx}_W_xg', W_xg)
            self.add_module(f'layer{layer_idx}_W_hg', W_hg)
            self.add_module(f'layer{layer_idx}_dropout', dropout_layer)

            # Store layer parameters
            self.layer_params.append((W_xz, W_hz, W_xr, W_hr, W_xg, W_hg, dropout_layer))

        # Output layer
        self.W_hy = nn.Linear(h_dim, out_dim, bias=True)
        # ========================

    def forward(self, input: Tensor, hidden_state: Tensor = None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        batch_size, seq_len, _ = input.shape

        layer_states = []
        for i in range(self.n_layers):
            if hidden_state is None:
                layer_states.append(
                    torch.zeros(batch_size, self.h_dim, device=input.device)
                )
            else:
                layer_states.append(hidden_state[:, i, :])

        layer_input = input
        layer_output = None

        # TODO: Implement the model's forward pass.
        #  Make sure to use the layer_params list you created.
        # ====== YOUR CODE: ======
        # Store all hidden states for each layer and timestep
        all_hidden_states = []

        # Process through each layer
        for layer_idx in range(self.n_layers):
            W_xz, W_hz, W_xr, W_hr, W_xg, W_hg, dropout = self.layer_params[layer_idx]

            # Initialize hidden states for this layer across all timesteps
            layer_hiddens = []
            h_t = layer_states[layer_idx]

            # Process each timestep for this layer
            for t in range(seq_len):
                x_t = layer_input[:, t, :]

                # GRU equations
                z_t = torch.sigmoid(W_xz(x_t) + W_hz(h_t))
                r_t = torch.sigmoid(W_xr(x_t) + W_hr(h_t))
                g_t = torch.tanh(W_xg(x_t) + W_hg(r_t * h_t))
                h_t = z_t * h_t + (1 - z_t) * g_t

                layer_hiddens.append(h_t)

            # Stack all hidden states for this layer: (S, B, H) -> (B, S, H)
            layer_hiddens = torch.stack(layer_hiddens, dim=1)

            # Apply dropout to the entire sequence
            layer_hiddens = dropout(layer_hiddens)

            # Update the final hidden state for this layer
            layer_states[layer_idx] = h_t

            # Use this layer's output as input to next layer
            layer_input = layer_hiddens

        # Final output projection for all timesteps at once
        # layer_input is now (B, S, H) from the last layer
        layer_output = self.W_hy(layer_input)  # (B, S, O)

        # Stack final hidden states for all layers
        hidden_state = torch.stack(layer_states, dim=1)  # (B, L, H)
        # ========================
        return layer_output, hidden_state
