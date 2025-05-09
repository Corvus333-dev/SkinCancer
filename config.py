from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class ModelConfig:
    def __post_init__(self):
        if self.mode not in {'train', 'dev', 'test'}:
            raise ValueError(f'Invalid mode: {self.mode}')

        if self.mode != 'train' and not self.checkpoint:
            raise ValueError('Model checkpoint required for dev/test modes.')

        if self.input_shape[-1] != 3:
            raise ValueError('Expected 3-channel RBG input.')

    framework: str
    mode: str
    checkpoint: Optional[str] = None
    augment: bool = True
    class_weight: bool = True
    dist_plot: bool = False
    freeze: bool = True
    learning_rate_decay: bool = True
    input_shape: Tuple[int, int, int] = (224, 224, 3)
    batch_size: int = 32
    dropout: float = 0.3
    l2_lambda: float = 0.01
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 30