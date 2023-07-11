import torch
import torch.nn as nn
import torch.nn.functional as F

from argument_parser import get_args

args = get_args()


class LSTM(nn.Module):
    """
    LSTM model
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_size,
        num_layers,
        output_size,
        dropout_rate,
    ):
        """
        Initialize the model
        :param vocab_size {int}: size of vocabulary
        :param embed_dim {int}: embedding dimension
        :param hidden_size {int}: hidden size
        :param num_layers {int}: number of layers
        :param output_size {int}: output size
        :param dropout_rate {float}: dropout rate
        """

        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(hidden_size, output_size)

    def forward(self, character, hidden, cell):
        """
        Forward pass
        :param character: input character
        :param hidden: hidden state
        :param cell: cell state
        Returns:
            output (torch.Tensor): output
            hidden (torch.Tensor): hidden state
            cell (torch.Tensor): cell state
        """

        output = self.embedding(character).unsqueeze(
            1
        )  # reshape to batch_size * 1 * embed_dim
        output, (hidden, cell) = self.lstm(output, (hidden, cell))
        output = self.dropout(output)  # applying dropout to the output
        output = F.relu(output)
        output = self.fc1(output).reshape(
            output.size(0), -1
        )  # reshape to batch_size * output_size

        return output, hidden, cell

    def init_hidden(self, batch_size):
        """
        Initialize hidden state
        :param batch_size: batch size
        Returns:
            hidden (torch.Tensor): hidden state
            cell (torch.Tensor): cell state
        """
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden, cell
