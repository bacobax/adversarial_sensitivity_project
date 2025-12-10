from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torchattacks as ta

# Torchattacks-based helpers with a small registry so adding new attacks stays simple.
# Each helper returns the adversarial images; the registry allows swapping/adding attacks later.

# Map lowercase attack names -> torchattacks class
ATTACK_REGISTRY: Dict[str, type] = {
    'fgsm': ta.FGSM,
    'pgd': ta.PGD,
    'deepfool': ta.DeepFool,
}


def available_attacks() -> Iterable[str]:
    """Return the list of supported attack names."""
    return sorted(ATTACK_REGISTRY.keys())


def create_attack(name: str, model: nn.Module, **kwargs) -> ta.Attack:
    """Instantiate a torchattacks Attack from the registry.

    Args:
        name: attack name (case-insensitive), e.g. "fgsm" or "pgd".
        model: classifier to attack.
        **kwargs: forwarded to the underlying torchattacks constructor.

    Raises:
        KeyError if the attack is not registered.
    """
    key = name.lower()
    if key not in ATTACK_REGISTRY:
        raise KeyError(f"Unsupported attack '{name}'. Available: {available_attacks()}")
    cls = ATTACK_REGISTRY[key]
    return cls(model, **kwargs)


def run_attack(
    name: str,
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    post_clamp: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> torch.Tensor:
    """Run an attack by name and return adversarial images."""
    atk = create_attack(name, model, **kwargs)
    adv = atk(images, labels)
    if post_clamp is not None:
        lo, hi = post_clamp
        adv = adv.clamp(float(lo), float(hi))
    return adv.detach()


def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 8 / 255,
    post_clamp: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> torch.Tensor:
    """FGSM attack wrapper (uses torchattacks.FGSM)."""
    return run_attack('fgsm', model, images, labels, eps=eps, post_clamp=post_clamp, **kwargs)


def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 8 / 255,
    alpha: float = 2 / 255,
    steps: int = 10,
    random_start: bool = True,
    post_clamp: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> torch.Tensor:
    """PGD attack wrapper (uses torchattacks.PGD)."""
    return run_attack(
        'pgd',
        model,
        images,
        labels,
        eps=eps,
        alpha=alpha,
        steps=steps,
        random_start=random_start,
        post_clamp=post_clamp,
        **kwargs,
    )


def deepfool_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    steps: int = 50,
    overshoot: float = 0.02,
    post_clamp: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> torch.Tensor:
    return run_attack(
        'deepfool',
        model,
        images,
        labels,
        steps=steps,
        overshoot=overshoot,
        post_clamp=post_clamp,
        **kwargs,
    )


__all__ = [
    'available_attacks',
    'create_attack',
    'run_attack',
    'fgsm_attack',
    'pgd_attack',
    'deepfool_attack',
]
