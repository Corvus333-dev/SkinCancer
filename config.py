from dataclasses import dataclass

@dataclass
class ModelConfig:
    framework: str
    input_shape: tuple[int, int, int]
    batch_size: int = 32
    freeze_layers: bool = True
    training: bool = False
    learning_rate: float = 0.001
    epochs: int = 10