import dataclasses
from typing import Literal, Optional, Tuple, Union

BASE_MODELS = {'efficientnetb1': (244, 244, 3), 'resnet50v2': (224, 224, 3)}

@dataclasses.dataclass
class ExpConfig:
    architecture: Literal['efficientnetb1', 'resnet50v2']
    mode: Literal['train', 'val', 'test']
    checkpoint: Optional[str] = None
    input_shape: tuple = dataclasses.field(init=False)
    freeze_bn: bool = False
    unfreeze: Optional[Union[int, str, Tuple[str, ...]]] = None

    def __post_init__(self):
        if self.architecture not in BASE_MODELS:
            raise ValueError(f'Invalid architecture: {self.architecture}')

        if self.mode not in {'train', 'val', 'test'}:
            raise ValueError(f'Invalid mode: {self.mode}')

        if self.mode != 'train' and not self.checkpoint:
            raise ValueError('Model checkpoint required for val/test modes.')

        self.input_shape = BASE_MODELS[self.architecture]

@dataclasses.dataclass
class TrainConfig:
    batch_size: int = 64
    boost: Optional[dict] = None
    dropout: Tuple[float, float, float] = (0.5, 0.25, 0.125)
    epochs: int = 50
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