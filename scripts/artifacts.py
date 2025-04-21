from datetime import datetime
from pathlib import Path
from dataclasses import asdict
import json
import pandas as pd
from sklearn.metrics import classification_report

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

def save_training_artifacts(directory, model, config, history, metrics):
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

def save_prediction_artifacts(dx_map, y, y_hat, config):
    """
    Saves actual vs predicted labels, and classification report.

    Args:
        dx_map (dict): Map of diagnosis codes to diagnosis names.
        y (list): True label indices.
        y_hat (np.ndarray): Predicted label indices.
        config (dataclass): Experiment configuration settings.

    Returns:
        None
    """
    df = pd.DataFrame({
        'actual': [dx_map[i] for i in y],
        'predicted': [dx_map[i] for i in y_hat]
    })

    directory = Path(config.checkpoint).parent

    if config.mode == 'validate':
        df_path = directory / 'dev_predictions.csv'
        report_path = directory / 'dev_report.json'
    elif config.mode == 'test':
        df_path = directory / 'test_predictions.csv'
        report_path = directory / 'test_report.json'

    df.to_csv(df_path, index=False)

    report = classification_report(y, y_hat, target_names=list(dx_map.values()), output_dict=True)

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)