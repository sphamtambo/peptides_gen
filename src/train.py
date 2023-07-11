#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    model,
    train_dl,
    optimizer,
    criterion,
    scheduler,
    weight_decay,
    batch_size,
    block_size,
):
    """
    train the model
    :param model: {torch.nn.Module} model to be trained
    :param train_dl: {torch.utils.data.DataLoader} training data loader
    :param optimizer: {torch.optim} optimizer
    :param criterion: {torch.nn} loss function
    :param scheduler: {torch.optim.lr_scheduler} learning rate scheduler
    :param weight_decay: {float} weight decay
    :param batch_size: {int} batch size
    :param block_size: {int} block size
    Returns:
        -train_loss: {float} training loss
        -train_acc: {float} training accuracy
    """

    model.train()
    train_running_loss = 0.0
    train_running_acc = 0

    text_batch, target_batch = next(iter(train_dl))
    # for text_batch, target_batch in train_dl:
    text_batch.to(device)
    target_batch.to(device)
    optimizer.zero_grad()
    loss = 0
    # forward pass
    hidden, cell = model.init_hidden(batch_size)
    for c in range(block_size):
        pred, hidden, cell = model(text_batch[:, c], hidden, cell)
        loss += criterion(pred, target_batch[:, c])

        # L2 regularization
        l2_loss = 0.0
        for param in model.parameters():
            l2_loss += torch.norm(param, p=2)
        loss += weight_decay * l2_loss

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    # update parameters
    optimizer.step()
    loss = loss.item() / block_size
    train_running_loss += loss
    # accuracy
    pred = torch.argmax(pred, dim=1)
    target = target_batch[:, c]
    acc = (pred == target).sum().item() / len(target)
    train_running_acc += acc
    scheduler.step()

    return train_running_loss, train_running_acc


def validate(model, val_dl, optimizer, criterion, batch_size, block_size):
    """
    validate the model
    :param model: {torch.nn.Module} model to be trained
    :param val_dl: {torch.utils.data.DataLoader} validation data loader
    :param optimizer: {torch.optim} optimizer
    :param criterion: {torch.nn} loss function
    :param batch_size: {int} batch size
    :param block_size: {int} block size
    Returns:
        -val_loss: {float} validation loss
        -val_acc: {float} validation accuracy
    """

    model.eval()
    with torch.no_grad():
        val_running_loss = 0.0
        val_running_acc = 0

        text_batch, target_batch = next(iter(val_dl))
        # for text_batch, target_batch in val_dl:
        text_batch.to(device)
        target_batch.to(device)
        optimizer.zero_grad()
        loss = 0
        # forward pass
        hidden, cell = model.init_hidden(batch_size)
        for c in range(block_size):
            pred, hidden, cell = model(text_batch[:, c], hidden, cell)
            loss += criterion(pred, target_batch[:, c])
        loss = loss.item() / block_size
        val_running_loss += loss
        # accuracy
        pred = torch.argmax(pred, dim=1)
        target = target_batch[:, c]
        acc = (pred == target).sum().item() / len(target)
        val_running_acc += acc

    return val_running_loss, val_running_acc
