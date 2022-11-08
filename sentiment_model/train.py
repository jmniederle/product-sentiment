from pathlib import Path
from typing import Dict
import random
import string
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from wandb.wandb_run import Run as RunLogger
from data_utils.metrics import accuracy, MAE
import numpy as np


def train(
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        epochs: int,
        logging_freq: int,
        logging: RunLogger,
        device: torch.device,
        checkpoint_path: Path,
        save_checkpoint: bool,
        num_classes: int,
        config: Dict,
):
    """Training loop with included validation at the end of each epoch.

    """
    model.to(device)  # In case it hasn't been sent to device yet
    p_bar = tqdm(total=len(train_loader.dataset) * epochs)

    if logging is not None:
        logging.watch(model, log="all", log_freq=logging_freq, log_graph=True)
        log_name = logging.name

    else:
        log_name = get_run_name(num_classes)
        # Calculate metrics
        logging_dict = {
            "train_loss": [],
            "valid_loss": [],
            "valid_score":[]
        }

    best_valid_loss = float('inf')
    last_valid_loss = float('inf')

    step = -1
    for epoch in range(epochs):
        # Training
        model.train()
        for batch_idx, (data, target, text_lengths) in enumerate(train_loader):
            step += 1

            data, target, text_lengths = data.to(device), target.to(device), text_lengths.to(device)

            # Forward pass
            output = model(data, text_lengths)

            if num_classes == 1:
                target = target.unsqueeze(1).float()

            loss = loss_fn(output, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate metrics
            log_dict = {"train_loss": loss.data}

            # Update logging and progress bar
            p_bar.set_description(f"Epoch: {epoch + 1} of {epochs} | "
                                  f"Training loss: {loss.data:.5f}"
                                  f" | Validation loss: {last_valid_loss:.5f}")
            p_bar.update(data.shape[0])

            if logging is not None:
                logging.log(log_dict, step=step)

            else:
                logging_dict['train_loss'].append(loss.data.item())

        # Validation
        model.eval()
        with torch.no_grad():

            valid_losses = []
            valid_scores = []

            for batch_idx, (data, target, text_lengths) in enumerate(valid_loader):

                data, target, text_lengths = data.to(device), target.to(device), text_lengths.to(device)

                # Forward pass
                output = model(data, text_lengths)

                if num_classes == 1:
                    target = target.unsqueeze(1).float()

                loss = loss_fn(output, target)
                valid_losses.append(loss.item())

                if num_classes > 1:
                    # Calculate metrics
                    acc = accuracy(output, target)
                    valid_scores.append(acc.item())

                elif num_classes == 1:

                    mae = MAE(output, target, apply_sigmoid=True)
                    valid_scores.append(mae.item())

            last_valid_loss = np.mean(valid_losses)

            log_dict = {"valid_loss": last_valid_loss, "valid_score": np.mean(valid_scores)}

            if logging is not None:
                # Log metrics
                logging.log(log_dict, step=step)

            else:
                logging_dict['valid_loss'].append(log_dict['valid_loss'])
                logging_dict['valid_score'].append(log_dict['valid_score'])

            if save_checkpoint and (last_valid_loss < best_valid_loss):

                # Save checkpoint if better than previous runs
                best_valid_loss = last_valid_loss

                # Validate checkpoint dir
                if not Path(checkpoint_path / log_name).exists():
                    Path(checkpoint_path / log_name).mkdir(parents=True)

                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "valid_loss": log_dict['valid_loss'],
                    "valid_score": log_dict['valid_score'],
                    "config": config,
                }, checkpoint_path / f"{log_name}" / f"{log_name}-epoch-{epoch}.pth")

            # if device == "cuda":
            #     torch.cuda.empty_cache()


def get_run_name(num_classes):
    rand_string = ''.join(random.choices(string.ascii_uppercase, k=5))
    run_name = f"run_{num_classes}_classes_{rand_string}"
    return run_name

# TODO: use different validation metric for sentiment140
