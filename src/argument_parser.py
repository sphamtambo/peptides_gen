import argparse


def get_args():
    """
    Get arguments from command line
    :return: args
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="sequences.csv",
        help="dataset file for training, a csv file",
    )
    parser.add_argument(
        "--random_seed", type=int, default=123, help="rando seed for reproducibility"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="the batch size to update network"
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=40,
        help="portion of characters used for training batch",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=20,
        help="number of unique characters or words used used as input features",
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=100,
        help="number of embedding demensions aka dimension vectors to represent entire vocab as input features",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=200,
        help="number of neurons to be projects from the input feature size",
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="number of hidden layers"
    )
    parser.add_argument(
        "--output_size",
        type=int,
        default=20,
        help="number of unique characters or words as output features",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.5,
        help="probability of an element to be zeroed in dropout layer",
    )
    parser.add_argument(
        "--l2_reg",
        type=float,
        default=0.03,
        help="l2 regularization coefficient",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.59,
        help="gamma for learning rate scheduler",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        help="momentum for optimizer learning rate used to update weights",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.79,
        help="alpha for optimizer learning rate used to update weights",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-08,
        help="epsilon for optimizer learning rate used to update weights",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="number of steps to accumulate gradients before performing a backward/update pass",
    )
    parser.add_argument(
        "--betas",
        type=float,
        default=(0.9, 0.999),
        help="beta1 for optimizer learning rate used to update weights",
    )
    parser.add_argument(
        "--step_size",
        type=float,
        default=50,
        help="step size for learning rate scheduler to decay learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.001,
        help="weight decay for optimizer learning rate used to update weights",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=200,
        help="number of epochs. Number of full passes through the training examples.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="learning rate for model training",
    )

    parser.add_argument(
        "--k_folds",
        type=int,
        default=3,
        help="cross validation folds for training",
    )
    parser.add_argument(
        "--loss_plot_name",
        type=str,
        default="loss.png",
        help="loss plot name to save the plot",
    )
    parser.add_argument(
        "--accuracy_plot_name",
        type=str,
        default="accuracy.png",
        help="accuracy plot name to save the plot",
    )
    parser.add_argument(
        "--sampled_text_name",
        type=str,
        default="sampled_text.txt",
        help="sampled text name to save the text",
    )
    parser.add_argument(
        "--mode",
        choices=["pretrain", "finetune", "sample"],
        default="pretrain",
        help="Mode: pretrain, finetune, sample",
    )
    parser.add_argument(
        "--session_name",
        default="pretrain",
        help="Mode: pretrain, finetune, sample",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="model",
        help="model name to save the model",
    )
    parser.add_argument(
        "--start_char",
        type=str,
        default="B",
        help="start character to begin sampling",
    )
    parser.add_argument(
        "--sample_len",
        type=int,
        default=100,
        help="number of sequences to sample training",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=1.0,
        help="temperature used to sample text",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        # default=None,
        help="filename of the pretrained model to used for sampling if train=False",
    )

    args = parser.parse_args()

    return args
