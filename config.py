import dataclasses
from typing import Literal, Optional, Tuple, Union

# Supported base models and input shapes
BASE_MODELS = {'efficientnetb1': (240, 240, 3), 'resnet50': (224, 224, 3), None: None}

@dataclasses.dataclass
class ExpConfig:
    """
    General experiment settings.

    Attributes:
        backbone: Base model architecture.
        mode: Operating mode (i.e., train/val/test/ensemble).
        checkpoint: Path to saved model. Required for 'val' or 'test' modes.
        input_shape: Image dimensions expected by backbone (auto set).
        unfreeze (int | str | tuple): Layers to unfreeze.
            - int: unfreeze from this depth upward
            - str: unfreeze from this layer name upward
            - tuple: unfreeze layers containing any of these keywords
        best_models: Paths to best models for ensembling (minimum of two required for 'ensemble' mode).

    Notes:
        - Input shape is set to ImageNet-pretrained resolution to ensure compatibility with pretrained weights.
    """
    mode: Literal['train', 'val', 'test', 'ensemble']
    backbone: Optional[Literal['efficientnetb1', 'resnet50']] = None
    checkpoint: Optional[str] = None
    input_shape: tuple = dataclasses.field(init=False)
    unfreeze: Optional[Union[int, str, Tuple[str, ...]]] = None
    best_models: Optional[Tuple[str, ...]] = None

    def __post_init__(self):
        if self.mode not in {'train', 'val', 'test', 'ensemble'}:
            raise ValueError(f'Invalid mode: {self.mode}')

        if self.mode != 'ensemble':
            if self.backbone is None or self.backbone not in BASE_MODELS:
                raise ValueError(f'Invalid backbone: {self.backbone}')

        if self.mode in ('val', 'test') and not self.checkpoint:
            raise ValueError('Model checkpoint required for val/test modes.')

        if self.mode == 'ensemble':
            if not self.best_models or len(self.best_models) < 2:
                raise ValueError('At least two models required for ensembling.')

        self.input_shape = BASE_MODELS[self.backbone]

@dataclasses.dataclass
class TrainConfig:
    """
    Training hyperparameters.

    Attributes:
        batch_size: Samples per batch.
        boost: Map of diagnosis codes to weight multipliers.
            Example: {0: 1.0, 1: 1.4, ..., 6: 1.2}
        dropout_rates: Dropout rates applied to each dense layer (bottom to top).
        epochs: Maximum training epochs.
        focal_loss: Parameters (alpha, gamma, label_smoothing) for sparse categorical focal cross-entropy loss.
            - alpha: inverse frequency weighting exponent (e.g., 0.5 = inverse sqrt, 1.0 = full inverse, etc.)
            - gamma: focusing parameter (e.g., 0.0 = standard cross-entropy, 2.0 = strong focusing, etc.)
            - label_smoothing: label smoothing factor (e.g., 0.0 = no smoothing, 0.1 = 10% smoothing, etc.)
        initial_lr: Starting learning rate.
        lr_decay: Use cosine decay.
        patience: Number of epochs with no improvement after which training will be stopped.
        weight_decay: Weight decay for optimizer.
    """
    batch_size: int = 64
    boost: Optional[dict] = None
    dropout_rates: Tuple[float, float, float] = (0.5, 0.25, 0.125)
    epochs: int = 100
    focal_loss: Optional[Tuple[float, float, float]] = None
    initial_lr: float = 1e-3
    lr_decay: bool = True
    patience: int = 10
    warmup_target: Optional[float] = None
    weight_decay: float = 1e-4

@dataclasses.dataclass
class Config:
    exp: ExpConfig
    train: TrainConfig