from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from wandb.wandb_run import Run as RunLogger
from data_utils.metrics import accuracy


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
        config: Dict,
):
    """Training loop with included validation at the end of each epoch.

    """
    model.to(device)  # In case it hasn't been sent to device yet
    p_bar = tqdm(total=len(train_loader.dataset) * epochs)

    logging.watch(model, log="all", log_freq=logging_freq, log_graph=True)

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

            logging.log(log_dict, step=step)

        # Validation
        model.eval()
        with torch.no_grad():

            for batch_idx, (data, target, text_lengths) in enumerate(valid_loader):
                # For assignment 3.2, we need to know the lengths of the targets

                data, target, text_lengths = data.to(device), target.to(device), text_lengths.to(device)

                # Forward pass
                output = model(data, text_lengths)
                loss = loss_fn(output, target)

                last_valid_loss = loss.data

                # Calculate metrics
                acc = accuracy(output, target)
                log_dict = {"valid_loss": loss.data, "valid_acc": acc}

                # Log metrics
                logging.log(log_dict, step=step)

                if save_checkpoint:

                    # Save checkpoint if better than previous runs
                    if loss < best_valid_loss:
                        best_valid_loss = loss

                        # Validate checkpoint dir
                        if not Path(checkpoint_path / logging.name).exists():
                            Path(checkpoint_path / logging.name).mkdir(parents=True)
                        torch.save({
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "epoch": epoch,
                            "train_metrics": None,
                            "valid_metrics": None,
                            "config": config,
                        }, checkpoint_path / f"{logging.name}" / f"{logging.name}-epoch-{epoch}.pth")

            if device == "cuda":
                torch.cuda.empty_cache()

