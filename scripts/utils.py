from dataclasses import asdict
from datetime import datetime
import json
import pandas as pd
from pathlib import Path

def save_dist(dist_plot):
    dist_path = 'data/distribution.png'
    dist_plot.savefig(dist_path, dpi=300)

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

def save_model(directory, model, config, history, hist_plot):
    """
    Saves trained model and its associated metadata.

    Args:
        directory (Path): Object pointing to experiment folder.
        model (keras.Model): Trained model.
        config (dataclass): Experiment configuration settings.
        history (keras.callbacks.History): Training history.
        hist_plot (matplotlib.figure.Figure): Training history plot.

    Returns:
        None
    """
    model.save(directory / 'model.keras')

    with open(directory / 'config.json', 'w') as f:
        json.dump(asdict(config), f, indent=4)

    with open(directory / 'summary.txt', 'w', encoding='utf-8') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    with open(directory / 'training_history.json', 'w') as f:
        json.dump(history.history, f, indent=4)

    hist_plot.savefig(directory / 'training_history.png', dpi=300)

def save_results(dx_map, y, y_hat, cr, cm_plot, metrics, config):
    """
    Saves evaluation metrics, predictions, and classification report.

    Args:
        dx_map (dict): Map of diagnosis codes to diagnosis names.
        y (list): True label indices.
        y_hat (np.ndarray): Predicted label indices.
        cr (dict): Classification report.
        cm_plot (matplotlib.figure.Figure): Confusion matrix plot.
        metrics (dict): Evaluation metrics.
        config (dataclass): Experiment configuration settings.

    Returns:
        None
    """
    df = pd.DataFrame({
        'actual': [dx_map[i] for i in y],
        'predicted': [dx_map[i] for i in y_hat]
    })

    directory = Path(config.checkpoint).parent

    if config.mode == 'dev':
        ev_path = directory / 'dev_evaluation_metrics.csv'
        df_path = directory / 'dev_predictions.csv'
        cr_path = directory / 'dev_classification_report.json'
        cm_path = directory / 'dev_confusion_matrix.png'
    elif config.mode == 'test':
        ev_path = directory / 'test_evaluation_metrics.csv'
        df_path = directory / 'test_predictions.csv'
        cr_path = directory / 'test_classification_report.json'
        cm_path = directory / 'test_confusion_matrix.png'

    # Save evaluation metrics
    with open(ev_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    # Save predictions
    df.to_csv(df_path, index=False)

    # Save classification report
    with open(cr_path, 'w') as f:
        json.dump(cr, f, indent=4)

    # Save confusion matrix
    cm_plot.savefig(cm_path, dpi=300)