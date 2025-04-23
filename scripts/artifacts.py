from datetime import datetime
from pathlib import Path
from dataclasses import asdict
import json
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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

    with open(directory / 'training_history.json', 'w') as f:
        json.dump(history.history, f, indent=4)

    with open(directory / 'evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

def save_prediction_artifacts(dx_map, y, y_hat, config):
    """
    Saves predictions, classification report, and confusion matrix.

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

    if config.mode == 'dev':
        df_path = directory / 'dev_predictions.csv'
        cr_path = directory / 'dev_classification_report.json'
        cm_path = directory / 'dev_confusion_matrix.png'
    elif config.mode == 'test':
        df_path = directory / 'test_predictions.csv'
        cr_path = directory / 'test_classification_report.json'
        cm_path = directory / 'test_confusion_matrix.png'

    # Save predictions
    df.to_csv(df_path, index=False)

    labels = list(dx_map.values())

    # Classification report
    cr = classification_report(y, y_hat, target_names=labels, output_dict=True)
    with open(cr_path, 'w') as f:
        json.dump(cr, f, indent=4)

    # Confusion matrix
    cm = confusion_matrix(y, y_hat)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap='Purples', ax=ax, xticks_rotation=45)
    plt.title(f'Confusion Matrix ({config.mode})')
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()