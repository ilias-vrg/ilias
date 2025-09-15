import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import BatchSampler, RandomSampler

from src.utils import get_progress_bar, AverageMeter


class LinearAdapter(nn.Module):
    def __init__(self, in_features, out_features):
        """Initialize a linear adapter layer.
        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
        """
        super(LinearAdapter, self).__init__()
        self.layer = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize the layer parameters."""
        nn.init.xavier_uniform_(self.layer.weight)
        if self.layer.bias is not None:
            nn.init.constant_(self.layer.bias, 0.0)

    def forward(self, x):
        """Forward pass through the linear adapter layer.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Normalized output tensor.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.layer(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


class ClassificationNetwork(nn.Module):
    """
    A simple classification network with a linear adaptation (projection) layer and a cosine similarity classifier head.

    - Projects input features to a new space via a learnable linear layer (the "adaptation layer").
    - Computes class logits as the cosine similarity between normalized features and normalized class weights.
    """

    def __init__(
        self,
        in_features: int,
        adaptation_layer_dim: int,
        num_classes: int,
    ) -> None:
        """
        Initializes the ClassificationNetwork.

        Args:
            in_features (int): Input feature dimensionality.
            adaptation_layer_dim (int): Output dimensionality of the adaptation layer.
            num_classes (int): Number of output classes.
        """
        super().__init__()
        self.adaptation_layer = LinearAdapter(
            in_features=in_features,
            out_features=adaptation_layer_dim,
        )
        self.classifier = nn.Linear(adaptation_layer_dim, num_classes, bias=False)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            batch (torch.Tensor): Input features of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Cosine similarity logits of shape (batch_size, num_classes).
        """
        # Project input to adaptation layer output
        adapted_batch = self.adaptation_layer(batch)

        # L2 normalize adapted features and classifier weights
        norm_adapted_batch = F.normalize(adapted_batch, dim=-1)
        norm_weight = F.normalize(self.classifier.weight, p=2, dim=1)

        # Compute cosine similarities (dot product after normalization)
        cosine_sims = torch.matmul(norm_adapted_batch, norm_weight.T)

        return cosine_sims  # These are the logits for classification


def train_lin_adapt_layer(
    dataset,
    adaptation_layer_dim: int = 512,
    num_epochs: int = 5,
    batch_size: int = 128,
    lr: float = 0.001,
    use_gpu: bool = True,
    verbose: int = 1,
    print_every: int = 50,
):
    """
    Trains a linear adaptation layer (first layer of a classification network) on provided dataset features.

    All data is used for training (no validation split).
    Inspired by TLDR (https://github.com/naver/tldr).

    Args:
        dataset (UnEDFeatureDataset): Dataset containing `.uned_feat` (features) and `.encoded_indices` (labels).
        adaptation_layer_dim (int): Output dimension of adaptation layer.
        num_epochs (int): Number of epochs to train for.
        batch_size (int): Mini-batch size.
        lr (float): Learning rate for Adam optimizer.
        use_gpu (bool): Whether to train on CUDA if available.
        verbose (int): If >0, show progress.
        print_every (int): Print loss every N steps.

    Returns:
        torch.nn.Module: The trained adaptation (linear) layer.
    """
    # Setup network
    classification_network = ClassificationNetwork(
        in_features=dataset.uned_feat.shape[1],
        adaptation_layer_dim=adaptation_layer_dim,
        num_classes=dataset.total_num_classes,
    )

    print(
        f"> Will train linear adaptation layer with {adaptation_layer_dim} output features"
    )

    # Load features and labels into memory (not suitable for very large datasets)
    X = dataset.uned_feat[:].astype(np.float32)  # (N, feature_dim) numpy
    X = torch.from_numpy(X).float()
    labels = torch.tensor(np.array(dataset.encoded_indices), dtype=torch.long)

    if use_gpu:
        X = X.cuda()
        labels = labels.cuda()
        classification_network = classification_network.cuda()

    n_data = X.shape[0]

    # Create batch sampler for random sampling
    batch_sampler = BatchSampler(
        RandomSampler(range(n_data)), batch_size=batch_size, drop_last=True
    )

    # Loss and optimizer
    loss_fn = SoftmaxMarginLoss(
        num_train_classes=dataset.total_num_classes,
        transform_logits_type="cosface",
        margin=0.0,
        scale=16.0,
    )
    optimizer = torch.optim.Adam(
        classification_network.parameters(), lr=lr, weight_decay=1e-6
    )
    losses = AverageMeter("Loss", ":.4e")

    step = 0

    with get_progress_bar() as progress:
        task = (
            progress.add_task(
                description="[green]Training Adaptation Layer",
                total=(len(batch_sampler) * num_epochs),
                info="-",
            )
            if verbose > 0
            else None
        )

        for epoch in range(num_epochs):
            if task is not None:
                progress.update(task, info=f"epoch {epoch + 1} (of {num_epochs})")

            for _, ind in enumerate(batch_sampler):
                step += 1

                feats = X[ind, :]
                batch_labels = labels[ind]

                optimizer.zero_grad()
                logits = classification_network(feats)
                loss = loss_fn(logits, batch_labels)

                losses.update(loss.item(), feats.size(0))
                loss.backward()
                optimizer.step()

                if print_every and step % print_every == 0 and verbose > 0:
                    progress.console.print(f" * {losses}, LR = {lr:.5f}")

                if task is not None:
                    progress.update(task, advance=1)

    return (
        classification_network.adaptation_layer
    )  # Return only the adaptation (linear) layer


class SoftmaxMarginLoss(nn.Module):
    """
    Implements a softmax margin loss, commonly used in deep metric learning tasks such as face recognition.
    This loss allows for margin-based logit transformations (e.g., ArcFace, CosFace) before applying softmax cross-entropy.

    Args:
        num_train_classes (int): Number of training classes for one-hot encoding.
        scale (float, optional): Scaling factor applied to logits. If `trainable_scale` is True, this becomes a learnable parameter.
        margin (float, optional): Margin added to the ground truth logit (ArcFace) or subtracted (CosFace).
        transform_logits_type (str, optional): If set, applies margin-based transformation. Choices: {"arcface", "cosface", None}.
        trainable_scale (bool, optional): If True, makes `scale` a trainable parameter.
    """

    def __init__(
        self,
        num_train_classes,
        scale=None,
        margin=None,
        transform_logits_type=None,
        trainable_scale=False,
    ):
        super().__init__()
        self.num_train_classes = num_train_classes
        self.margin = margin
        self.scale = scale
        self.transform_logits_type = transform_logits_type
        self.loss_fn = nn.CrossEntropyLoss()

        # Optionally make the scale parameter trainable
        if trainable_scale:
            if self.scale is None:
                raise ValueError("scale must be specified if trainable_scale=True")
            self.scale = nn.Parameter(torch.tensor(self.scale, dtype=torch.float32))

    def _transform_logits(self, logits, labels):
        """
        Optionally transforms logits with a margin-based method before loss computation.

        Args:
            logits (Tensor): Raw logits of shape (batch_size, num_train_classes).
            labels (Tensor): Integer labels of shape (batch_size,).

        Returns:
            Tensor: Transformed logits of the same shape as input.
        """
        if self.transform_logits_type is None:
            return logits

        one_hot = F.one_hot(labels, num_classes=self.num_train_classes).float()

        if self.transform_logits_type == "arcface":
            # For ArcFace: logits are assumed to be cos(theta), margin is added in angular space
            theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
            target_logits = torch.cos(theta + self.margin)
            logits = target_logits * one_hot + logits * (1 - one_hot)

        elif self.transform_logits_type == "cosface":
            # For CosFace: margin is subtracted from the target logit directly
            logits = (logits - self.margin) * one_hot + logits * (1 - one_hot)

        return logits

    def forward(self, logits, labels):
        """
        Applies (optional) logit transformation and computes cross-entropy loss.

        Args:
            logits (Tensor): Input logits (batch_size, num_train_classes).
            labels (Tensor): Ground-truth integer labels (batch_size,).

        Returns:
            Tensor: Loss value (scalar).
        """
        logits = self._transform_logits(logits, labels)
        # Optionally apply scaling
        if self.scale is not None:
            logits = logits * self.scale
        return self.loss_fn(logits, labels)
