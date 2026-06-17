# training/train.py

import copy
from training.evaluate import evaluate_model


def train_one_model(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    device,
    num_epochs,
    patience,
):
    """Train model with early stopping and return best state and metrics."""

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(num_epochs):
        model.train()

        total_train_loss = 0.0
        total_batches = 0

        for images, targets in train_loader:
            # Move batch to device
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).float()

            # Forward + backward + update
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_batches += 1

        avg_train_loss = total_train_loss / max(1, total_batches)
        train_losses.append(avg_train_loss)

        # Evaluate on validation set
        avg_val_loss, _ = evaluate_model(
            model=model,
            criterion=criterion,
            loader=val_loader,
            device=device,
        )
        val_losses.append(avg_val_loss)

        print(
            f"Epoch {epoch + 1:>2}/{num_epochs} | "
            f"train loss={avg_train_loss:.4f} | "
            f"val loss={avg_val_loss:.4f}"
        )

        # Track best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping condition
        if epochs_without_improvement >= patience:
            stopped_early = True
            print(f"Early stopped at epoch {epoch + 1}")
            break

    if not stopped_early:
        print(f"Finished {num_epochs} epochs")

    return (
        best_state,
        train_losses,
        val_losses,
        best_val_loss,
        best_epoch,
        stopped_early,
    )