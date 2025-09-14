from dataclasses import asdict
from datetime import datetime
import json
import pandas as pd
from pathlib import Path

def save_dist(dist_plot):
    """
    Saves plot of diagnosis frequencies occurring in HAM10000 dataset.

    Args:
        dist_plot (matplotlib.figure.Figure): Distribution plot.

    Returns:
        None
    """
    dist_path = 'data/distribution.png'
    dist_plot.savefig(dist_path, dpi=300)

def create_directory(backbone):
    """
    Creates a backbone-specific, timestamped directory for storing experiment results.

    Args:
        backbone (str): Base model architecture.

    Returns:
        Path: Object pointing to new directory.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    directory = Path('models') / backbone / timestamp
    directory.mkdir(parents=True, exist_ok=True)

    return directory

def save_model(directory, model, config, history, hist_plot, layer_state):
    """
    Saves trained model and its associated metadata.

    Args:
        directory (Path): Object pointing to experiment folder.
        model (keras.Model): Trained model.
        config (dataclass): Experiment configuration settings.
        history (keras.callbacks.History): Training history.
        hist_plot (matplotlib.figure.Figure): Training history plot.
        layer_state (dict): Map of layer names and training states.

    Returns:
        None
    """
    # model.save(directory / 'model.keras')

    with open(directory / 'config.json', 'w') as f:
        json.dump(asdict(config), f, indent=4)

    with open(directory / 'summary.txt', 'w', encoding='utf-8') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    with open(directory / 'layer_state.json', 'w') as f:
        json.dump(layer_state, f, indent=4)

    with open(directory / 'training_history.json', 'w') as f:
        json.dump(history.history, f, indent=4)

    hist_plot.savefig(directory / 'training_history.png', dpi=300)

def save_results(dx_map, y, y_hat, cr, cm_plot, prc_data, prc_plot, checkpoint, mode):
    """
    Saves predictions, classification report, confusion matrix, and precision-recall curve.

    Args:
        dx_map (dict): Map of diagnosis codes to diagnosis names.
        y (list): True diagnosis indices.
        y_hat (np.ndarray): Predicted diagnosis indices.
        cr (dict): Classification report.
        cm_plot (matplotlib.figure.Figure): Confusion matrix plot.
        prc_data (dict): Precision-recall curve data.
        prc_plot (matplotlib.figure.Figure): Precision-recall curve plot.
        checkpoint (str): Location of saved model (used to extract parent directory).
        mode (str): mode (str): Validation or test mode designation.

    Returns:
        None
    """
    df = pd.DataFrame({
        'actual': [dx_map[i] for i in y],
        'predicted': [dx_map[i] for i in y_hat]
    })

    directory = Path(checkpoint).parent

    if mode == 'val':
        df_path = directory / 'val_predictions.csv'
        cr_path = directory / 'val_classification_report.json'
        cm_path = directory / 'val_confusion_matrix.png'
        prc_data_path = directory / 'val_prc.json'
        prc_plot_path = directory / 'val_prc.png'
    elif mode == 'test':
        df_path = directory / 'test_predictions.csv'
        cr_path = directory / 'test_classification_report.json'
        cm_path = directory / 'test_confusion_matrix.png'
        prc_data_path = directory / 'test_prc.json'
        prc_plot_path = directory / 'test_prc.png'
    else:
        raise AssertionError('Mode validation should be handled by ExperimentConfig.')

    # Save predictions
    df.to_csv(df_path, index=False)

    # Save classification report
    with open(cr_path, 'w') as f:
        json.dump(cr, f, indent=4)

    # Save confusion matrix
    cm_plot.savefig(cm_path, dpi=300)

    # Save precision-recall curve data
    with open(prc_data_path, 'w') as f:
        json.dump(prc_data, f, indent=4)

    # Save precision-recall curve plot
    prc_plot.savefig(prc_plot_path, dpi=300)