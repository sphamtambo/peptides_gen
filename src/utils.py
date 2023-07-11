import glob as glob
import os

import matplotlib.pyplot as plt
import torch
from torch.distributions.categorical import Categorical

from dataset import string_to_int, vocab


def save_plots(
    train_accs,
    valid_accs,
    train_losses,
    valid_losses,
    acc_plot_path,
    loss_plot_path,
):
    """
    Function to save the loss and accuracy plots to disk.
    :param train_accs: {list} list of training accuracies
    :param valid_accs: {list} list of validation accuracies
    :param train_losses: {list} list of training losses
    :param valid_losses: {list} list of validation losses
    :param acc_plot_path: {str} path to save the accuracy plot
    :param loss_plot_path: {str} path to save the loss plot
    Returns:
        -None
    """

    # Accuracy plots.
    plt.figure(figsize=(10, 7))
    plt.plot(train_accs, color="green", linestyle="-", label="train accuracy")
    plt.plot(valid_accs, color="blue", linestyle="-", label="validataion accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(acc_plot_path)

    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(train_losses, color="orange", linestyle="-", label="train loss")
    plt.plot(valid_losses, color="red", linestyle="-", label="validataion loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(loss_plot_path)


def sample(model, starting_str, num_chars, scale_factor):
    """
    Function to sample text from the trained model.
    :param model: {torch.nn.Module} the trained model
    :param starting_str: {str} the starting string for the model to generate text
    :param num_chars: {int} the number of characters to generate
    :param scale_factor: {int} the scale factor to scale the logits
    :param num_texts: {int} the number of text to generate
    Returns:
        -generated_text: {str} the generated text
    """

    # sample after every training epoch
    encoded_input = string_to_int(starting_str)
    encoded_input = encoded_input.unsqueeze(0)

    generated_text = starting_str

    model.eval()
    hidden, cell = model.init_hidden(1)
    for c in range(len(starting_str) - 1):
        _, hidden, cell = model(encoded_input[:, c].view(1), hidden, cell)

    last_char = encoded_input[:, -1]
    for _ in range(num_chars):
        logits, hidden, cell = model(last_char.view(1), hidden, cell)
        logits = logits.squeeze(0)

        scaled_logits = logits * scale_factor
        m = Categorical(logits=scaled_logits)
        last_char = m.sample()
        generated_text += vocab[last_char]

    return generated_text


def save_generated(sampled, filename):
    """
    Function to save the generated text to disk.
    :param sampled: {str} the generated text
    :param filename: {str} the path to save the generated text
    Returns:
        -None
    """

    with open(filename, "w") as f:
        f.write(sampled + "\n")


def save_model(model, path):
    """
    Function to save the model to disk.
    :param model: model state dict to save
    :param path: {str} the path to save the model
    Returns:
        -saved model in checkpoint dir
    """
    torch.save(model, path)


def load_model(model, path):
    """
    Function to load the model from disk.
    :param model: {torch.nn.Module} the model to load
    :param path: {str} the path to load the model
    Returns:
        -None
    """
    model.load_state_dict(torch.load(path))
