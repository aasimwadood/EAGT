"""
Focal Loss and class weight utilities for handling class imbalance.

This module implements:
- FocalLoss: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
- Class weight computation based on inverse frequency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in affect recognition.


    Focal Loss down-weights well-classified examples and focuses on hard,
    misclassified examples. This is particularly useful for affect datasets
    where engagement typically dominates over frustration or boredom.

    Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Parameters
    ----------
    alpha : torch.Tensor, optional
        Class weights tensor of shape (num_classes,). If None, all classes
        are weighted equally.
    gamma : float
        Focusing parameter. Higher values focus more on hard examples.
        Default is 2.0 as commonly used in literature.
    reduction : str
        Specifies the reduction to apply: 'none', 'mean', 'sum'.
    label_smoothing : float
        Label smoothing factor for regularization. Default 0.0.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Parameters
        ----------
        inputs : torch.Tensor
            Predicted logits of shape (B, C) where B is batch size and C is num_classes.
        targets : torch.Tensor
            Ground truth class indices of shape (B,).

        Returns
        -------
        torch.Tensor
            Computed focal loss (scalar if reduction is 'mean' or 'sum').
        """
        # Compute cross entropy loss (without reduction)
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing
        )

        # Compute p_t (probability of correct class)
        pt = torch.exp(-ce_loss)

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - pt) ** self.gamma

        # Apply class weights if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_weight = alpha_t * focal_weight

        # Compute final loss
        loss = focal_weight * ce_loss

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy loss with label smoothing.

    Useful as an alternative to focal loss for regularization.
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)

        # Create smoothed targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = (-smooth_targets * log_probs).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def compute_class_weights(
    labels: torch.Tensor,
    num_classes: int = 4,
    method: str = "inverse_freq",
    beta: float = 0.9999,
) -> torch.Tensor:
    """
    Compute class weights for handling imbalanced datasets.


    Parameters
    ----------
    labels : torch.Tensor
        All labels in the dataset.
    num_classes : int
        Number of classes. Default 4 for EAGT (frustration, confusion, boredom, engagement).
    method : str
        Method for computing weights:
        - "inverse_freq": Simple inverse frequency
        - "effective_num": Effective number of samples (better for long-tailed)
        - "sqrt_inverse": Square root of inverse frequency (gentler)
    beta : float
        Beta parameter for effective number method.

    Returns
    -------
    torch.Tensor
        Class weights tensor of shape (num_classes,).
    """
    # Count samples per class
    counts = torch.bincount(labels.view(-1).long(), minlength=num_classes).float()

    # Avoid division by zero
    counts = counts + 1e-6

    if method == "inverse_freq":
        # Simple inverse frequency
        weights = 1.0 / counts

    elif method == "effective_num":
        # Effective number of samples (from "Class-Balanced Loss")
        effective_num = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / effective_num

    elif method == "sqrt_inverse":
        # Square root of inverse frequency (gentler weighting)
        weights = 1.0 / torch.sqrt(counts)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize weights so they sum to num_classes
    weights = weights / weights.sum() * num_classes

    return weights


def compute_class_weights_from_dataloader(
    dataloader,
    num_classes: int = 4,
    label_key: str = "label",
    method: str = "inverse_freq",
) -> torch.Tensor:
    """
    Compute class weights by iterating through a dataloader.

    Parameters
    ----------
    dataloader : DataLoader
        PyTorch dataloader.
    num_classes : int
        Number of classes.
    label_key : str
        Key to access labels in batch dict.
    method : str
        Weighting method (see compute_class_weights).

    Returns
    -------
    torch.Tensor
        Class weights tensor.
    """
    all_labels = []
    for batch in dataloader:
        if isinstance(batch, dict):
            labels = batch[label_key]
        else:
            labels = batch[1]  # Assume (data, labels) tuple
        all_labels.append(labels)

    all_labels = torch.cat(all_labels)
    return compute_class_weights(all_labels, num_classes, method)


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining classification with auxiliary objectives.

    multitask learning with
    arousal/valence prediction alongside categorical states.

    Parameters
    ----------
    classification_weight : float
        Weight for main classification loss.
    auxiliary_weight : float
        Weight for auxiliary regression loss (arousal/valence).
    focal_gamma : float
        Gamma for focal loss on classification.
    """

    def __init__(
        self,
        classification_weight: float = 1.0,
        auxiliary_weight: float = 0.3,
        focal_gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.classification_weight = classification_weight
        self.auxiliary_weight = auxiliary_weight
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        class_logits: torch.Tensor,
        class_targets: torch.Tensor,
        aux_preds: Optional[torch.Tensor] = None,
        aux_targets: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute combined loss.

        Returns
        -------
        dict with keys: 'total', 'classification', 'auxiliary'
        """
        # Classification loss
        cls_loss = self.focal_loss(class_logits, class_targets)

        # Auxiliary loss (if provided)
        aux_loss = torch.tensor(0.0, device=class_logits.device)
        if aux_preds is not None and aux_targets is not None:
            aux_loss = self.mse_loss(aux_preds, aux_targets)

        # Combined loss
        total_loss = (
            self.classification_weight * cls_loss +
            self.auxiliary_weight * aux_loss
        )

        return {
            "total": total_loss,
            "classification": cls_loss,
            "auxiliary": aux_loss,
        }
