"""
Training script for EAGT Affect Recognition.

Paper Reference: Section III.D

This module implements:
- Training loop with focal loss for class imbalance (Section III.D.6)
- Data augmentation (Section III.D.7)
- Hybrid fusion model with auxiliary losses (Section III.D.2)
- Learning rate warmup (100 steps)
- Early stopping
- Leave-one-subject-out evaluation support

Usage:
    python -m eagt.training.train_affect --config configs/train.yaml
"""

import os
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

from eagt.data.datasets import EAGTDataset
from eagt.models.fusion import FusionClassifier, FusionClassifierHybrid, AuxiliaryLoss
from eagt.training.losses import FocalLoss, compute_class_weights
from eagt.data.augmentation import MultimodalAugmentation, get_augmentation_pipeline


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    loss: float = 0.0
    accuracy: float = 0.0
    f1_macro: float = 0.0
    f1_per_class: Optional[Dict[str, float]] = None


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Paper Reference: Section III.D mentions early stopping for training.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "max",
    ):
        """
        Parameters
        ----------
        patience : int
            Number of epochs to wait before stopping.
        min_delta : float
            Minimum change to qualify as an improvement.
        mode : str
            "min" for loss, "max" for accuracy/F1.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Returns True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def _make_dataloaders(cfg) -> Dict[str, DataLoader]:
    """
    Builds train/val dataloaders.

    Paper Reference: Section III.D mentions leave-one-subject-out evaluation.
    For proper evaluation, use separate train/val CSV files.
    """
    # Get CSV paths
    train_csv = getattr(cfg.data, 'train_csv', cfg.data.split_csv)
    val_csv = getattr(cfg.data, 'val_csv', cfg.data.split_csv)

    # Create datasets
    ds_train = EAGTDataset(
        train_csv,
        use_real_features=getattr(cfg.data, 'use_real_features', False),
    )
    ds_val = EAGTDataset(
        val_csv,
        use_real_features=getattr(cfg.data, 'use_real_features', False),
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=getattr(cfg.train, 'num_workers', 0),
        drop_last=True,
        pin_memory=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=getattr(cfg.train, 'num_workers', 0),
        drop_last=False,
        pin_memory=True,
    )
    return {"train": dl_train, "val": dl_val}


def create_optimizer_and_scheduler(
    model: nn.Module,
    cfg,
    num_training_steps: int,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """
    Create optimizer with warmup scheduler.

    Paper Reference: Section III.D mentions LR warmup (100 steps).
    """
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    # Warmup scheduler (100 steps per paper)
    warmup_steps = getattr(cfg.train, 'warmup_steps', 100)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    # Main scheduler (cosine annealing)
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps - warmup_steps,
        eta_min=cfg.train.lr * 0.01,
    )

    # Combined scheduler
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps],
    )

    return optimizer, scheduler


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    aux_criterion: Optional[nn.Module],
    augmentation: Optional[nn.Module],
    device: torch.device,
    cfg,
    use_hybrid: bool = True,
) -> TrainingMetrics:
    """
    Train for one epoch.

    Paper Reference: Section III.D

    Uses:
    - Focal loss for class imbalance (Section III.D.6)
    - Data augmentation (Section III.D.7)
    - Auxiliary losses for hybrid fusion (Section III.D.2)
    """
    model.train()
    if augmentation:
        augmentation.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Move to device
        face = batch["face"].to(device)
        audio = batch["audio"].to(device)
        behav = batch["behav"].to(device)
        labels = batch["label"].to(device)

        # Apply augmentation
        if augmentation:
            aug_batch = augmentation({
                "face": face,
                "audio": audio,
                "behav": behav,
                "label": labels,
            })
            face = aug_batch["face"]
            audio = aug_batch["audio"]
            behav = aug_batch["behav"]

        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        if use_hybrid:
            outputs = model(face, audio, behav)
            logits = outputs["logits"]

            # Compute main loss
            main_loss = criterion(logits, labels)

            # Compute auxiliary losses (if using hybrid model)
            if aux_criterion is not None:
                aux_losses = aux_criterion(outputs, labels)
                loss = aux_losses["total"]
            else:
                loss = main_loss
        else:
            logits = model(face, audio, behav)
            loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg.train.grad_clip,
        )

        optimizer.step()
        scheduler.step()

        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        # Update progress bar
        acc = (preds == labels).float().mean().item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.3f}")

    # Compute epoch metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return TrainingMetrics(
        loss=avg_loss,
        accuracy=accuracy,
        f1_macro=f1,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list,
    use_hybrid: bool = True,
    silent: bool = False,
) -> TrainingMetrics:
    """
    Evaluate model on validation set.
    """
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = dataloader if silent else tqdm(dataloader, desc="Evaluating")
    for batch in pbar:
        face = batch["face"].to(device)
        audio = batch["audio"].to(device)
        behav = batch["behav"].to(device)
        labels = batch["label"].to(device)

        # Forward pass
        if use_hybrid:
            outputs = model(face, audio, behav)
            logits = outputs["logits"]
        else:
            logits = model(face, audio, behav)

        loss = criterion(logits, labels)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    if not silent:
        report = classification_report(
            all_labels,
            all_preds,
            target_names=class_names,
            digits=4,
        )
        print(f"\n{report}")

    # Per-class F1
    f1_per_class = {}
    for i, name in enumerate(class_names):
        class_preds = [1 if p == i else 0 for p in all_preds]
        class_labels = [1 if l == i else 0 for l in all_labels]
        if sum(class_labels) > 0:
            f1_per_class[name] = f1_score(class_labels, class_preds)

    return TrainingMetrics(
        loss=avg_loss,
        accuracy=accuracy,
        f1_macro=f1,
        f1_per_class=f1_per_class,
    )


def train_loop(cfg) -> None:
    """
    Main training loop.

    Paper Reference: Section III.D

    Implements:
    - Focal loss for class imbalance (Section III.D.6)
    - Data augmentation (Section III.D.7)
    - Hybrid fusion with auxiliary losses (Section III.D.2)
    - LR warmup (100 steps)
    - Early stopping
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataloaders
    dls = _make_dataloaders(cfg)

    # Determine model type
    use_hybrid = getattr(cfg.model, 'use_hybrid', True)

    # Create model
    if use_hybrid:
        model = FusionClassifierHybrid(
            face_dim=getattr(cfg.model, 'face_dim', 512),
            audio_dim=getattr(cfg.model, 'audio_dim', 768),
            behav_dim=getattr(cfg.model, 'behav_dim', 16),
            hidden=getattr(cfg.model, 'hidden', 256),
            num_layers=getattr(cfg.model, 'num_layers', 2),
            num_classes=len(cfg.data.classes),
            dropout=getattr(cfg.model, 'dropout', 0.30),
            learn_fusion_weights=True,
        ).to(device)
    else:
        model = FusionClassifier(
            num_classes=len(cfg.data.classes),
        ).to(device)

    print(f"Model: {'FusionClassifierHybrid' if use_hybrid else 'FusionClassifier'}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Compute class weights for focal loss
    use_focal_loss = getattr(cfg.train, 'use_focal_loss', True)
    if use_focal_loss:
        # Collect all labels to compute weights
        all_labels = []
        for batch in dls["train"]:
            all_labels.extend(batch["label"].tolist())
        all_labels = torch.tensor(all_labels)

        class_weights = compute_class_weights(
            all_labels,
            num_classes=len(cfg.data.classes),
            method="effective_num",
        ).to(device)

        criterion = FocalLoss(
            alpha=class_weights,
            gamma=getattr(cfg.train, 'focal_gamma', 2.0),
        )
        print(f"Using Focal Loss with gamma={cfg.train.focal_gamma}")
        print(f"Class weights: {class_weights.cpu().tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using CrossEntropyLoss")

    # Auxiliary loss for hybrid model
    aux_criterion = None
    if use_hybrid:
        aux_criterion = AuxiliaryLoss(
            main_weight=1.0,
            aux_weight=getattr(cfg.train, 'aux_weight', 0.3),
            criterion=criterion,
        )
        print(f"Using auxiliary losses with weight={cfg.train.aux_weight}")

    # Data augmentation
    use_augmentation = getattr(cfg.train, 'use_augmentation', True)
    augmentation = None
    if use_augmentation:
        augmentation = get_augmentation_pipeline(
            audio_noise_std=getattr(cfg.augment, 'audio_noise_std', 0.01),
            time_mask_prob=getattr(cfg.augment, 'time_mask_prob', 0.1),
            modality_dropout_prob=getattr(cfg.augment, 'modality_dropout_prob', 0.1),
        )
        print("Using data augmentation")

    # Create optimizer and scheduler with warmup
    num_training_steps = len(dls["train"]) * cfg.train.epochs
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, cfg, num_training_steps
    )
    print(f"LR warmup: {getattr(cfg.train, 'warmup_steps', 100)} steps")

    # Early stopping
    early_stopping = EarlyStopping(
        patience=getattr(cfg.train, 'early_stopping_patience', 10),
        mode="max",  # Maximize F1 score
    )
    print(f"Early stopping patience: {early_stopping.patience}")

    # Create checkpoint directory
    os.makedirs(cfg.train.ckpt_dir, exist_ok=True)

    # Training loop
    best_f1 = 0.0
    best_epoch = 0

    for epoch in range(1, cfg.train.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{cfg.train.epochs}")
        print(f"{'='*60}")

        # Train
        train_metrics = train_epoch(
            model=model,
            dataloader=dls["train"],
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            aux_criterion=aux_criterion,
            augmentation=augmentation,
            device=device,
            cfg=cfg,
            use_hybrid=use_hybrid,
        )

        print(f"Train - Loss: {train_metrics.loss:.4f}, "
              f"Acc: {train_metrics.accuracy:.4f}, "
              f"F1: {train_metrics.f1_macro:.4f}")

        # Evaluate
        val_metrics = evaluate(
            model=model,
            dataloader=dls["val"],
            criterion=criterion,
            device=device,
            class_names=cfg.data.classes,
            use_hybrid=use_hybrid,
            silent=True,
        )

        print(f"Val   - Loss: {val_metrics.loss:.4f}, "
              f"Acc: {val_metrics.accuracy:.4f}, "
              f"F1: {val_metrics.f1_macro:.4f}")

        if use_hybrid:
            weights = model.get_fusion_weights()
            print(f"Fusion weights: {weights.cpu().tolist()}")

        # Save checkpoint
        ckpt_path = os.path.join(cfg.train.ckpt_dir, f"model_epoch{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }, ckpt_path)

        # Save best model
        if val_metrics.f1_macro > best_f1:
            best_f1 = val_metrics.f1_macro
            best_epoch = epoch
            best_path = os.path.join(cfg.train.ckpt_dir, "model_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
            }, best_path)
            print(f"New best model saved! F1: {best_f1:.4f}")

        # Early stopping check
        if early_stopping(val_metrics.f1_macro):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            print(f"Best F1: {best_f1:.4f} at epoch {best_epoch}")
            break

    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best validation F1: {best_f1:.4f} at epoch {best_epoch}")
    print(f"Best model saved to: {cfg.train.ckpt_dir}/model_best.pt")


@torch.no_grad()
def evaluate_checkpoint(cfg, ckpt_path: str, silent: bool = False) -> TrainingMetrics:
    """
    Load a checkpoint and evaluate on validation set.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    dls = _make_dataloaders(cfg)

    # Determine model type
    use_hybrid = getattr(cfg.model, 'use_hybrid', True)

    # Create model
    if use_hybrid:
        model = FusionClassifierHybrid(num_classes=len(cfg.data.classes))
    else:
        model = FusionClassifier(num_classes=len(cfg.data.classes))

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()

    criterion = nn.CrossEntropyLoss()

    metrics = evaluate(
        model=model,
        dataloader=dls["val"],
        criterion=criterion,
        device=device,
        class_names=cfg.data.classes,
        use_hybrid=use_hybrid,
        silent=silent,
    )

    return metrics


def leave_one_subject_out_cv(cfg) -> Dict[str, float]:
    """
    Leave-one-subject-out cross-validation.

    Paper Reference: Section III.D mentions this evaluation strategy.

    This requires a CSV with a 'subject_id' column to identify subjects.
    """
    import pandas as pd

    # Load full dataset
    df = pd.read_csv(cfg.data.split_csv)

    if 'subject_id' not in df.columns:
        raise ValueError("CSV must have 'subject_id' column for LOSO-CV")

    subjects = df['subject_id'].unique()
    print(f"Running LOSO-CV with {len(subjects)} subjects")

    all_results = []

    for held_out_subject in subjects:
        print(f"\n{'='*40}")
        print(f"Holding out subject: {held_out_subject}")
        print(f"{'='*40}")

        # Create train/val splits
        train_df = df[df['subject_id'] != held_out_subject]
        val_df = df[df['subject_id'] == held_out_subject]

        # Save temporary CSVs
        train_csv = os.path.join(cfg.train.ckpt_dir, "train_temp.csv")
        val_csv = os.path.join(cfg.train.ckpt_dir, "val_temp.csv")
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)

        # Update config
        cfg.data.train_csv = train_csv
        cfg.data.val_csv = val_csv

        # Train
        train_loop(cfg)

        # Get best model results
        best_ckpt = os.path.join(cfg.train.ckpt_dir, "model_best.pt")
        metrics = evaluate_checkpoint(cfg, best_ckpt, silent=True)

        all_results.append({
            'subject': held_out_subject,
            'accuracy': metrics.accuracy,
            'f1_macro': metrics.f1_macro,
        })

        # Cleanup
        os.remove(train_csv)
        os.remove(val_csv)

    # Aggregate results
    import numpy as np
    accuracies = [r['accuracy'] for r in all_results]
    f1_scores = [r['f1_macro'] for r in all_results]

    results = {
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'mean_f1': np.mean(f1_scores),
        'std_f1': np.std(f1_scores),
    }

    print(f"\n{'='*60}")
    print("LOSO-CV Results:")
    print(f"Accuracy: {results['mean_accuracy']:.4f} +/- {results['std_accuracy']:.4f}")
    print(f"F1 Macro: {results['mean_f1']:.4f} +/- {results['std_f1']:.4f}")
    print(f"{'='*60}")

    return results
