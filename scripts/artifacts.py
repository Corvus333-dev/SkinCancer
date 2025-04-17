from datetime import datetime
from pathlib import Path
from dataclasses import asdict
import json

def create_directory(framework):
    """
    Creates a model-specific directory for storing experiment results.

    Args:
        framework (str): Base model architecture.

    Returns:
        Path: Object pointing to new directory.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    directory = Path('models') / f'{framework}_{timestamp}'
    directory.mkdir(parents=True, exist_ok=True)

    return directory

def save_artifacts(directory, model, config, history, metrics):
    """
    Saves trained model and its associated metadata.

    Args:
        directory (Path): Object pointing to experiment folder.
        model (keras.Model): Trained model.
        config (dataclass): Experiment configuration settings.
        history (keras.callbacks.History): Training history.
        metrics (dict): Evaluation metrics.

    Returns:
        None
    """
    model.save(directory / 'model.keras')

    with open(directory / 'config.json', 'w') as f:
        json.dump(asdict(config), f, indent=4)

    with open(directory / 'summary.txt', 'w', encoding='utf-8') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    with open(directory / 'history.json', 'w') as f:
        json.dump(history.history, f, indent=4)

    with open(directory / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)