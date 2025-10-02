from dataclasses import asdict
from datetime import datetime
import json
import pandas as pd
from pathlib import Path

def save_eda(age_dist_plot, dx_dist_plot):
    """
    Saves exploratory data analysis plots.

    Args:
        age_dist_plot (matplotlib.figure.Figure): Age distribution by diagnosis plot.
        dx_dist_plot (matplotlib.figure.Figure): Diagnosis (class) distribution plot.

    Returns:
        None
    """
    data_dir = Path('../data')
    age_dist_plot.savefig(data_dir / 'age_distribution.png', dpi=300)
    dx_dist_plot.savefig(data_dir / 'dx_distribution.png', dpi=300)

def make_exp_dir(name):
    """
    Creates a customizable-named, timestamped directory for storing experiment results.

    Args:
        name (str): Descriptive name for mid-level directory (e.g., 'resnet50', 'ensemble', etc.).

    Returns:
        Path: Object pointing to new directory (models/name/YYYYMMDD_HHMM).
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    exp_dir = Path('models') / name / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)

    return exp_dir

def save_model(model, config, history, hist_plot, layer_state, exp_dir):
    """
    Saves trained model and its artifacts.

    Args:
        model (keras.Model): Trained model.
        config (dataclass): Experiment configuration settings.
        history (keras.callbacks.History): Training history.
        hist_plot (matplotlib.figure.Figure): Training history plot.
        layer_state (dict): Map of layer names and training states.
        exp_dir (Path): Object pointing to experiment folder.

    Returns:
        None
    """
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(asdict(config), f, indent=4)

    with open(exp_dir / 'summary.txt', 'w', encoding='utf-8') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    with open(exp_dir / 'layer_state.json', 'w') as f:
        json.dump(layer_state, f, indent=4)

    with open(exp_dir / 'training_history.json', 'w') as f:
        json.dump(history.history, f, indent=4)

    hist_plot.savefig(exp_dir / 'training_history.png', dpi=300)

def save_results(pred_df, cr, cm_plot, prc_data, prc_plot, mode, exp_dir):
    """
    Saves predictions, classification report, confusion matrix, and precision-recall curve.

    Args:
        pred_df (pd.DataFrame): DataFrame containing prediction results (image_id, dx probability, actual, predicted).
        cr (dict): Classification report.
        cm_plot (matplotlib.figure.Figure): Confusion matrix plot.
        prc_data (dict): Precision-recall curve data.
        prc_plot (matplotlib.figure.Figure): Precision-recall curve plot.
        mode (str): mode (str): Mode setting (val/test/ensemble).
        exp_dir (Path): Object pointing to experiment folder.

    Returns:
        None
    """
    pred_df_path = exp_dir / f'{mode}_predictions.csv'
    cr_path = exp_dir / f'{mode}_classification_report.json'
    cm_path = exp_dir / f'{mode}_confusion_matrix.png'
    prc_data_path = exp_dir / f'{mode}_prc.json'
    prc_plot_path = exp_dir / f'{mode}_prc.png'

    # Save predictions
    pred_df.to_csv(pred_df_path, index=False)

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