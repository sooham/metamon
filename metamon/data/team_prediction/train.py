import os
import argparse
import random

import tqdm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import wandb

from metamon.data.team_prediction.dataset import (
    TeamPredictionDataset,
    CompetitiveTeamPredictionDataset,
)
from metamon.data.team_prediction.model import TeamTransformer
from metamon.data.team_prediction.vocabulary import Vocabulary
from metamon.data.team_prediction.team import TeamSet


def train(config, use_wandb: bool = True):
    # config: hyperparameters namespace or wandb.config

    # Set random seed
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Prepare datasets
    train_dset = TeamPredictionDataset(
        data_dir=config.train_data_dir,
        split="train",
        validation_ratio=config.val_ratio,
        mask_pokemon_prob_range=(config.mask_p, config.mask_p),
        mask_attrs_prob_range=(config.mask_a, config.mask_a),
        seed=config.seed,
    )
    val_dset = TeamPredictionDataset(
        data_dir=config.train_data_dir,
        split="val",
        validation_ratio=config.val_ratio,
        mask_pokemon_prob_range=(config.mask_p, config.mask_p),
        mask_attrs_prob_range=(config.mask_a, config.mask_a),
        seed=config.seed,
    )
    comp_dset = CompetitiveTeamPredictionDataset(
        mask_pokemon_prob_range=(config.mask_p, config.mask_p),
        mask_attrs_prob_range=(config.mask_a, config.mask_a),
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    comp_loader = DataLoader(
        comp_dset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    # Initialize model
    model = TeamTransformer(
        max_seq_len=config.max_seq_len,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_ff,
        dropout=config.dropout,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    vocab = Vocabulary()

    # Create checkpoint directory (per run) and artifacts subdir
    ckpt_dir = os.path.join(config.checkpoint_dir, config.run_name)
    artifact_dir = os.path.join(ckpt_dir, "artifacts")
    os.makedirs(artifact_dir, exist_ok=True)

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_count = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for x_tokens, type_ids, y_tokens in tqdm.tqdm(train_loader, desc="Training"):
            x_tokens = x_tokens.to(device)
            type_ids = type_ids.to(device)
            y_tokens = y_tokens.to(device)

            logits = model(x_tokens, type_ids)
            # Compute loss
            loss = F.cross_entropy(
                logits.view(-1, len(vocab.tokenizer)), y_tokens.view(-1)
            )
            # Compute accuracy
            preds = logits.argmax(dim=-1)
            train_correct += (preds == y_tokens).sum().item()
            train_total += y_tokens.numel()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        train_acc = train_correct / train_total if train_total else 0.0
        if use_wandb:
            wandb.log(
                {"replay_train/loss": train_loss, "replay_train/accuracy": train_acc},
                step=epoch,
            )
        else:
            print(
                f"Epoch {epoch} | Replay Train - loss: {train_loss:.4f}, accuracy: {train_acc:.4f}"
            )

        # Validation on replay_val split
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x_tokens, type_ids, y_tokens in tqdm.tqdm(
                val_loader, desc="Validation on replay_val split"
            ):
                x_tokens = x_tokens.to(device)
                type_ids = type_ids.to(device)
                y_tokens = y_tokens.to(device)
                logits = model(x_tokens, type_ids)
                loss = F.cross_entropy(
                    logits.view(-1, len(vocab.tokenizer)), y_tokens.view(-1)
                )
                # accuracy
                preds = logits.argmax(dim=-1)
                val_correct += (preds == y_tokens).sum().item()
                val_total += y_tokens.numel()
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total if val_total else 0.0
        if use_wandb:
            wandb.log(
                {"replay_val/loss": val_loss, "replay_val/accuracy": val_acc},
                step=epoch,
            )
        else:
            print(
                f"Epoch {epoch} | Replay Val   - loss: {val_loss:.4f}, accuracy: {val_acc:.4f}"
            )

        # Validation on competitive_val set
        comp_loss = 0.0
        comp_correct = 0
        comp_total = 0
        with torch.no_grad():
            for x_tokens, type_ids, y_tokens in tqdm.tqdm(
                comp_loader, desc="Validation on competitive_val set"
            ):
                x_tokens = x_tokens.to(device)
                type_ids = type_ids.to(device)
                y_tokens = y_tokens.to(device)
                logits = model(x_tokens, type_ids)
                loss = F.cross_entropy(
                    logits.view(-1, len(vocab.tokenizer)), y_tokens.view(-1)
                )
                # accuracy
                preds = logits.argmax(dim=-1)
                comp_correct += (preds == y_tokens).sum().item()
                comp_total += y_tokens.numel()
                comp_loss += loss.item()
        comp_loss /= len(comp_loader)
        comp_acc = comp_correct / comp_total if comp_total else 0.0
        if use_wandb:
            wandb.log(
                {
                    "competitive_val/loss": comp_loss,
                    "competitive_val/accuracy": comp_acc,
                },
                step=epoch,
            )
        else:
            print(
                f"Epoch {epoch} | Competitive - loss: {comp_loss:.4f}, accuracy: {comp_acc:.4f}"
            )

        # Log example predictions
        examples = next(iter(val_loader))
        x_tokens, type_ids, y_tokens = examples
        x_tokens = x_tokens.to(device)
        type_ids = type_ids.to(device)
        logits = model(x_tokens, type_ids)
        probs = torch.softmax(logits, dim=-1)
        filt = vocab.filter_probs(probs, type_ids)
        bs, seq_len, vs = filt.shape
        flat = filt.view(-1, vs)
        sampled = torch.multinomial(flat, 1).view(bs, seq_len)

        table = wandb.Table(columns=["input", "predicted", "ground_truth"])
        for i in range(min(bs, config.num_examples)):
            x_seq = vocab.ints_to_pokeset_seq(x_tokens[i].cpu().tolist())
            pred_seq = vocab.ints_to_pokeset_seq(sampled[i].tolist())
            true_seq = vocab.ints_to_pokeset_seq(y_tokens[i].cpu().tolist())
            x_team = TeamSet.from_seq(x_seq, include_stats=False)
            pred_team = TeamSet.from_seq(pred_seq, include_stats=False)
            true_team = TeamSet.from_seq(true_seq, include_stats=False)
            table.add_data(
                f"**Input**:\n{x_team.to_str()}",
                f"**Predicted**:\n{pred_team.to_str()}",
                f"**Ground truth**:\n{true_team.to_str()}",
            )
        if use_wandb:
            wandb.log({"predictions": table}, step=epoch)
        else:
            print(f"Examples at epoch {epoch}:")
            for row in table.data:
                print("---")
                print(row[0])
                print(row[1])
                print(row[2])

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            best_model = os.path.join(ckpt_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model)
            if use_wandb:
                # Log best checkpoint as Artifact
                artifact = wandb.Artifact(f"{config.run_name}-best-model", type="model")
                artifact.add_file(best_model)
                wandb.log_artifact(artifact)
            else:
                print(f"Best model saved to {best_model}")
        else:
            patience_count += 1
            if patience_count >= config.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Save final model
    final_model = os.path.join(ckpt_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model)
    if use_wandb:
        # Log final model as Artifact
        artifact = wandb.Artifact(f"{config.run_name}-final-model", type="model")
        artifact.add_file(final_model)
        wandb.log_artifact(artifact)
    else:
        print(f"Final model saved to {final_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TeamTransformer with optional W&B"
    )
    parser.add_argument("--project", type=str, help="W&B project name")
    parser.add_argument("--entity", type=str, help="W&B entity/user")
    parser.add_argument(
        "--group", type=str, default=None, help="W&B group name for sweeps"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging and print to console instead",
    )
    parser.add_argument(
        "--name", type=str, default=None, help="Run name to use for checkpoints and W&B"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )
    args = parser.parse_args()

    # Default hyperparameters
    sweep_defaults = {
        "train_data_dir": "/mnt/data1/shared_pokemon_project/metamon_team_files",
        "val_ratio": 0.1,
        "batch_size": 32,
        "num_workers": 0,
        "mask_p": 0.1,
        "mask_a": 0.1,
        "seed": 42,
        "max_seq_len": 64,
        "d_model": 256,
        "nhead": 4,
        "num_layers": 3,
        "dim_ff": 1024,
        "dropout": 0.1,
        "learning_rate": 1e-4,
        "epochs": 20,
        "patience": 5,
        "weight_decay": 1e-4,
        "num_examples": 4,
    }

    # Determine whether to use WandB
    use_wandb = not args.no_wandb
    if use_wandb:
        # Initialize WandB run
        wandb.init(
            project=args.project,
            entity=args.entity,
            group=args.group,
            config=sweep_defaults,
            name=args.name,
        )
        cfg = wandb.config
        # Override checkpoint dir & run name in config
        cfg.checkpoint_dir = args.checkpoint_dir
        cfg.run_name = wandb.run.name
    else:
        # Use local config namespace
        from argparse import Namespace

        cfg = Namespace(**sweep_defaults)
        cfg.checkpoint_dir = args.checkpoint_dir
        cfg.run_name = args.name or "local_run"

    # Start training
    train(cfg, use_wandb)
