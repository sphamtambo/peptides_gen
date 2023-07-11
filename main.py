#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings

warnings.filterwarnings(
    "ignore", message="A NumPy version .* is required for this version of SciPy"
)
import os
import sys
import time

sys.path.append("./src/")
sys.path.append("./models/")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from argument_parser import get_args
from dataset import (chars, reset_random_seeds, text, train_data,
                     train_dataloader, val_data, val_dataloader, vocab)
from models import LSTM
from train import train, validate
from utils import sample, save_generated, save_plots


def main():
    args = get_args()
    reset_random_seeds(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nDATA STATISTICS\n")
    print(f"Total samples: {len(text.split()):,}")
    print(f"Mean sample length: {np.mean([len(s) for s in text.split()]):.2f}")
    print(f"Standard deviation: {np.std([len(s) for s in text.split()]):.2f}")
    print(f"Max sample length: {np.max([len(s) for s in text.split()]):,}")
    print(f"Min sample length: {np.min([len(s) for s in text.split()]):,}")
    print(f"Vocabulary size: {len(chars):,}")
    print(f"Vocabulary: {vocab}")
    print("=" * 80)

    model = LSTM(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=args.vocab_size,
        dropout_rate=args.dropout_rate,
    )
    model.to(device)

    optimizer = optim.RMSprop(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        alpha=args.alpha,
        eps=args.epsilon,
        momentum=args.momentum,
        centered=False,
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )

    print("\nTRAINING INFO\n")
    print(f"Computing on: {device} device")
    print(f"Model architecture: {model}")
    print(f"Total samples: {len(text):,}")
    print(f"Total training samples: {len(train_data):,}")
    print(f"Total validation samples: {len(val_data):,}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Training parameters: {total_trainable_params:,}")
    print("=" * 80)

    session_name = args.session_name
    session_dir = os.path.join("reports", session_name)
    if not os.path.exists(session_dir):
        os.makedirs(session_dir)

    if args.mode == "pretrain":
        print("\nTRAINING MODEL")

        train_loss, train_acc = [], []
        val_loss, val_acc = [], []

        start_time = time.time()

        best_val_loss = np.inf
        best_model_state = None

        for epoch in range(args.num_epochs):
            train_epoch_loss, train_epoch_acc = train(
                model=model,
                train_dl=train_dataloader,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                block_size=args.block_size,
            )

            val_epoch_loss, val_epoch_acc = validate(
                model=model,
                val_dl=val_dataloader,
                optimizer=optimizer,
                criterion=criterion,
                batch_size=args.batch_size,
                block_size=args.block_size,
            )

            train_loss.append(train_epoch_loss)
            train_acc.append(train_epoch_acc)
            val_loss.append(val_epoch_loss)
            val_acc.append(val_epoch_acc)

            if epoch % 10 == 0:
                print(f"\nEpoch [{epoch+1}/{args.num_epochs}]")
                print(
                    f"Train Loss: {train_epoch_loss:.5f} | Train Accuracy: {train_epoch_acc:.2f}"
                )
                print(
                    f"Valid Loss: {val_epoch_loss:.5f} | Valid Accuracy: {val_epoch_acc:.2f}"
                )
                print("=" * 80)

                # save model
                if val_epoch_loss < best_val_loss:
                    best_val_loss = val_epoch_loss
                    best_model_state = model.state_dict()

        torch.save(
            best_model_state,
            f"reports/{session_name}/model_best.h5",
        )

        print("\nTRAINING COMPLETED")
        print(f"Total training time: {(time.time() - start_time)/60:.2f} min")

        save_plots(
            train_acc,
            val_acc,
            train_loss,
            val_loss,
            f"reports/{session_name}/{args.accuracy_plot_name}",
            f"reports/{session_name}/{args.loss_plot_name}",
        )

        print("\nGENERATING TEXT")
        generated = sample(
            model,
            starting_str=args.start_char,
            num_chars=args.sample_len,
            scale_factor=args.temp,
        )

        print("\nSAVING GENERATED TEXT")
        save_generated(generated, f"reports/{session_name}/sampled.txt")

    elif args.mode == "sample":
        print("\nGENERATING TEXT")
        # load pretrained model from checkpoint
        model.load_state_dict(torch.load(args.checkpoint))
        generated = sample(
            model,
            starting_str=args.start_char,
            num_chars=args.sample_len,
            scale_factor=args.temp,
        )

        print("\nSAVING GENERATED TEXT")
        save_generated(generated, f"reports/{session_name}/sampled.txt")

    elif args.mode == "finetune":
        print("\nFINETUNING MODEL")
        # load pretrained model from checkpoint
        model.load_state_dict(torch.load(args.checkpoint))

        start_time = time.time()

        train_loss, train_acc = [], []
        val_loss, val_acc = [], []

        best_val_loss = np.inf
        best_model_state = None

        for epoch in range(args.num_epochs):
            train_epoch_loss, train_epoch_acc = train(
                model=model,
                train_dl=train_dataloader,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                block_size=args.block_size,
            )

            val_epoch_loss, val_epoch_acc = validate(
                model=model,
                val_dl=val_dataloader,
                optimizer=optimizer,
                criterion=criterion,
                batch_size=args.batch_size,
                block_size=args.block_size,
            )

            train_loss.append(train_epoch_loss)
            train_acc.append(train_epoch_acc)
            val_loss.append(val_epoch_loss)
            val_acc.append(val_epoch_acc)

            if epoch % 10 == 0:
                print(f"\nEpoch [{epoch+1}/{args.num_epochs}]")
                print(
                    f"Train Loss: {train_epoch_loss:.5f} | Train Accuracy: {train_epoch_acc:.2f}"
                )
                print(
                    f"Valid Loss: {val_epoch_loss:.5f} | Valid Accuracy: {val_epoch_acc:.2f}"
                )
                print("=" * 80)

                # save model
                if val_epoch_loss < best_val_loss:
                    best_val_loss = val_epoch_loss
                    best_model_state = model.state_dict()

        torch.save(
            best_model_state,
            f"reports/{session_name}/model_best.h5",
        )

        print("\nFINETUNING COMPLETED")
        print(f"Total validation time: {(time.time() - start_time)/60:.2f} min")

        print("\nSAVING MODEL")

        save_plots(
            train_acc,
            val_acc,
            train_loss,
            val_loss,
            f"reports/{session_name}/{args.accuracy_plot_name}",
            f"reports/{session_name}/{args.loss_plot_name}",
        )

        print("\nGENERATING TEXT")
        generated = sample(
            model,
            starting_str=args.start_char,
            num_chars=args.sample_len,
            scale_factor=args.temp,
        )

        print("\nSAVING GENERATED TEXT")
        save_generated(generated, f"reports/{session_name}/sampled.txt")

    else:
        raise ValueError("Invalid mode selected. Please try again.")


if __name__ == "__main__":
    main()
