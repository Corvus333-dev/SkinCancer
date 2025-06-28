from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

@dataclass
class ExperimentConfig:
    architecture: Literal['efficientnetb0', 'inception_v3', 'resnet50']
    mode: Literal['train', 'dev', 'test']
    checkpoint: Optional[str] = None
    unfreeze: Optional[Union[int, str, Tuple[str, ...]]] = None
    boost: Optional[dict] = None
    class_weight: Optional[float] = None
    dist_plot: bool = False
    focal_loss: Optional[Tuple[float, float, float]] = None
    lr_decay: bool = True
    input_shape: Tuple[int, int, int] = (224, 224, 3)
    batch_size: int = 32
    dropout: Tuple[float, float, float, float] = (4, 3, 2, 1)
    initial_lr: float = 1e-3
    patience: int = 5
    warmup_target: Optional[float] = None
    weight_decay: float = 1e-5
    epochs: int = 30

    def __post_init__(self):
        if self.architecture not in {'efficientnetb0', 'inception_v3', 'resnet50'}:
            raise ValueError(f'Invalid architecture: {self.architecture}')

        if self.mode not in {'train', 'dev', 'test'}:
            raise ValueError(f'Invalid mode: {self.mode}')

        if self.mode != 'train' and not self.checkpoint:
            raise ValueError('Model checkpoint required for dev/test modes.')

        if self.input_shape[-1] != 3:
            raise ValueError('Expected 3-channel RBG input.')