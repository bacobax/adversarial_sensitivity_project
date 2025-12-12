from dataclasses import dataclass


@dataclass
class SamplePaths:
    """Container for all image paths related to a single sample."""
    filename: str
    real: str
    samecat: str
    diffcat: str
    mask_samecat: str
    mask_diffcat: str  # Derived from bbox
