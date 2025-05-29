from dataclasses import asdict
from datetime import datetime
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score, precision_recall_curve

def save_dist(dist_plot):
    dist_path = 'data/distribution.png'
    dist_plot.savefig(dist_path, dpi=300)

def create_directory(architecture):
    """
    Creates a model-specific directory for storing experiment results.

    Args:
        architecture (str): Base model architecture.

    Returns:
        Path: Object pointing to new directory.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    directory = Path('models') / f'{architecture}_{timestamp}'
    directory.mkdir(parents=True, exist_ok=True)

    return directory

def get_layer_state(model, architecture):
    """
    Extracts layer names and associated training states (unfrozen or frozen) from the base model.

    Args:
        model (keras.Model): Base model object.
        architecture (str): Base model architecture.

    Returns:
        dict: Map of layer names and training states.
    """
    base_model = model.get_layer(architecture)
    layer_state = {}

    for layer in base_model.layers:
        state = 'unfrozen' if layer.trainable else 'frozen'
        layer_state[layer.name] = state

    return layer_state

def get_prc_data(dx_names, p, y, classes=7):
    """
    Computes precision, recall, thresholds, and average precision values for each diagnosis class.

    Args:
        dx_names (list): Diagnosis names.
        p (np.ndarray): Probability distributions.
        y (np.ndarray): True diagnosis indices.
        classes (int): Number of classes.

    Returns:
        dict: Precision-recall curve values for each class.
    """
    prc_data = {}

    for i in range(classes):
        y_true = np.equal(y, i)
        y_score = p[:, i]

        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        avg_precision = average_precision_score(y_true, y_score)

        prc_data[dx_names[i]] = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': thresholds.tolist(),
            'avg_precision': avg_precision
        }

    return prc_data

def save_model(directory, model, config, layer_state, history, hist_plot):
    """
    Saves trained model and its associated metadata.

    Args:
        directory (Path): Object pointing to experiment folder.
        model (keras.Model): Trained model.
        config (dataclass): Experiment configuration settings.
        layer_state (dict): Map of layer names and training states.
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

    with open(directory / 'layer_state.json', 'w') as f:
        json.dump(layer_state, f, indent=4)

    with open(directory / 'training_history.json', 'w') as f:
        json.dump(history.history, f, indent=4)

    hist_plot.savefig(directory / 'training_history.png', dpi=300)

def save_results(dx_map, y, y_hat, cr, cm_plot, prc_data, config):
    """
    Saves predictions, classification report, confusion matrix, and precision-recall curve data.

    Args:
        dx_map (dict): Map of diagnosis codes to diagnosis names.
        y (list): True diagnosis indices.
        y_hat (np.ndarray): Predicted diagnosis indices.
        cr (dict): Classification report.
        cm_plot (matplotlib.figure.Figure): Confusion matrix plot.
        prc_data (dict): Precision-recall curve data.
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
        df_path = directory / 'dev_predictions.csv'
        cr_path = directory / 'dev_classification_report.json'
        cm_path = directory / 'dev_confusion_matrix.png'
        prc_path = directory / 'dev_prc.json'
    elif config.mode == 'test':
        df_path = directory / 'test_predictions.csv'
        cr_path = directory / 'test_classification_report.json'
        cm_path = directory / 'test_confusion_matrix.png'
        prc_path = directory / 'test_prc.json'

    # Save predictions
    df.to_csv(df_path, index=False)

    # Save classification report
    with open(cr_path, 'w') as f:
        json.dump(cr, f, indent=4)

    # Save confusion matrix
    cm_plot.savefig(cm_path, dpi=300)

    # Save precision-recall curve data
    with open(prc_path, 'w') as f:
        json.dump(prc_data, f, indent=4)