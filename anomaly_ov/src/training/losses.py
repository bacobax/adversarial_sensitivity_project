"""
Loss functions for Anomaly OV fine-tuning.
"""

import torch
import torch.nn as nn


class AnomalyStage1Loss(nn.Module):
    """
    Stage-1 loss aligned with paper intent:
    - BCE-with-logits on global anomaly score
    - Temperature scaling on logits (tau)
    - Class weighting via pos_weight for imbalance
    - Optional margin to push positives higher and negatives lower
    """
    
    def __init__(self, tau: float = 1.0, pos_weight: float = 1.0, margin: float = 0.0):
        """
        Args:
            tau: Temperature scaling factor for logits
            pos_weight: Weight for positive class (anomaly)
            margin: Margin to push positives higher and negatives lower
        """
        super().__init__()
        self.tau = tau
        self.margin = margin
        # Register pos_weight as buffer to follow the module device
        self.register_buffer('pos_weight', torch.tensor([pos_weight], dtype=torch.float32))
        # Placeholder; actual BCEWithLogitsLoss will be built on the fly to match device/dtype
        self._loss = None

    @staticmethod
    def to_logits(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Convert probability p in (0,1) to logit safely."""
        p = p.clamp(eps, 1 - eps)
        return torch.log(p / (1 - p))

    def _get_loss(self, device, dtype):
        """Create loss object on the correct device/dtype."""
        if (self._loss is None) or \
           (self._loss.weight is not None and self._loss.weight.device != device) or \
           (self.pos_weight.device != device):
            # Move buffer to device
            self.pos_weight = self.pos_weight.to(device=device, dtype=dtype)
            self._loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        return self._loss

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss.
        
        Args:
            probs: Model probabilities in [0,1], shape [B, 1]
            targets: Binary labels {0,1}, shape [B, 1]
        
        Returns:
            Loss value
        """
        # Convert to logits and apply temperature scaling
        logits = self.to_logits(probs) / self.tau
        
        # Apply class-dependent margin: shift logits for tighter separation
        if self.margin > 0.0:
            # Increase positive logits, decrease negative logits
            logits = logits + self.margin * (2 * targets - 1)  # +m for y=1, -m for y=0
        
        loss_fn = self._get_loss(device=logits.device, dtype=logits.dtype)
        loss = loss_fn(logits, targets)
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            probs: Model probabilities in [0,1], shape [B, 1]
            targets: Binary labels {0,1}, shape [B, 1]
        
        Returns:
            Loss value
        """
        eps = 1e-6
        probs = probs.clamp(eps, 1 - eps)
        
        # Get probability of true class
        p_t = targets * probs + (1 - targets) * (1 - probs)
        
        # Compute alpha weighting
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute loss
        loss = -alpha_t * focal_weight * torch.log(p_t)
        
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss with BCE and optional mask supervision.
    """
    
    def __init__(
        self, 
        tau: float = 1.0, 
        pos_weight: float = 1.0, 
        margin: float = 0.0,
        mask_weight: float = 0.0
    ):
        """
        Args:
            tau: Temperature scaling for BCE
            pos_weight: Weight for positive class
            margin: Margin for BCE
            mask_weight: Weight for mask loss (0 to disable)
        """
        super().__init__()
        self.bce_loss = AnomalyStage1Loss(tau=tau, pos_weight=pos_weight, margin=margin)
        self.mask_weight = mask_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self, 
        probs: torch.Tensor, 
        targets: torch.Tensor,
        pred_masks: torch.Tensor = None,
        gt_masks: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            probs: Model probabilities in [0,1], shape [B, 1]
            targets: Binary labels {0,1}, shape [B, 1]
            pred_masks: Predicted anomaly masks (optional)
            gt_masks: Ground truth masks (optional)
        
        Returns:
            Loss value
        """
        loss = self.bce_loss(probs, targets)
        
        if self.mask_weight > 0 and pred_masks is not None and gt_masks is not None:
            mask_loss = self.mse_loss(pred_masks, gt_masks)
            loss = loss + self.mask_weight * mask_loss
        
        return loss
