from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class ModelConfig:
    framework: str
    mode: str
    checkpoint: Optional[str] = None
    input_shape: Tuple[int, int, int] = (224, 224, 3)
    batch_size: int = 32
    freeze_layers: bool = True
    training: bool = False
    learning_rate: float = 0.001
    epochs: int = 10