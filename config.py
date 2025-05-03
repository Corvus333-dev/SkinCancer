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
    input_shape: Tuple[int, int, int] = (224, 224, 3)
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 10