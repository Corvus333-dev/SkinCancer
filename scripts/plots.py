import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

def plot_dist(df, dx_names):
    """
    Plots distribution of images used for diagnosis classification.

    Args:
        df (pd.DataFrame): DataFrame for HAM10000 dataset.
        dx_names (list): Alphabetized diagnosis names.

    Returns:
        matplotlib.figure.Figure: Distribution plot.
    """
    counts = df['dx'].value_counts()

    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    sns.barplot(x=counts.index, y=counts.values, ax=ax, color='cyan', order=dx_names)
    bars = ax.containers[0]
    ax.bar_label(bars, padding=3, fontsize=12)
    ax.set_xlabel('Diagnosis')
    ax.set_ylabel('Number of Images')
    ax.set_title('HAM10000 Dataset Distribution')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    fig.tight_layout()

    return fig

def plot_hist(history, directory):
    """
    Plots training history with dedicated axes for accuracy and loss.

    Args:
        history (dict): Training history metrics.
        directory (Path): Object pointing to experiment folder.

    Returns:
        matplotlib.figure.Figure: Training history plot.
    """
    exp_name = directory.name

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(history['accuracy'], color='magenta')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_title(f'Training Accuracy ({exp_name})')

    ax[1].plot(history['loss'], color='cyan')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].set_title(f'Training Loss ({exp_name})')

    fig.tight_layout()

    return fig

def plot_cm(cm, dx_names, config):
    """
    Plots a confusion matrix with normalized prediction rates.

    Args:
        cm (np.ndarray): Confusion matrix.
        dx_names (list): Diagnosis names.
        config (dataclass): Experiment configuration settings.

    Returns:
        matplotlib.figure.Figure: Confusion matrix plot.
    """
    exp_name = Path(config.checkpoint).parent.name
    mode = config.mode

    # Normalize values
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    # Customize annotations
    annot = np.empty_like(cm, dtype=object)
    length = cm.shape[0]
    for i in range(length):
        for j in range(length):
            annot[i, j] = f'{cm_norm[i, j]:.0f}%'

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.heatmap(
        cm_norm,
        annot=annot,
        ax=ax,
        cbar_kws={'label': 'Prediction Rate'},
        cmap='cool',
        fmt='',
        xticklabels=dx_names,
        yticklabels=dx_names
    )
    ax.set_xlabel('Predicted Diagnosis')
    ax.set_ylabel('Actual Diagnosis')
    ax.set_title(f'Normalized Confusion Matrix ({exp_name}, {mode})')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    fig.tight_layout()

    return fig