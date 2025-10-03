import matplotlib
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

def plot_age_dist(df, dx_names):
    """
    Plots age distribution by diagnosis as a 1.5 IQR boxplot.

    Note: unnormalize ages and exclude sentinel ages (0.0) before calling.

        Args:
            df (pd.dataFrame): DataFrame containing 'dx' and 'age' columns.
            dx_names (list): Diagnosis names.

        Returns:
            matplotlib.figure.Figure: Age distribution by diagnosis plot.
    """
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    sns.boxplot(
        data=df,
        x='dx',
        y='age',
        order=dx_names,
        color='cyan',
        width=0.5,
        fliersize=0.0,
        ax=ax
    )

    ax.set_xlabel('Diagnosis')
    ax.set_ylabel('Age (years)')
    ax.set_title('Age Distribution by Diagnosis')
    ax.tick_params(axis='x', labelrotation=45)

    fig.tight_layout()

    return fig

def plot_dx_dist(exp_counts, dropped, used, dx_names):
    """
    Plots diagnosis (class) distributions after undersampling as a bar chart. Includes an overlay for dropped samples.

    Args:
        exp_counts (pd.Series): Diagnosis value counts after undersampling.
        dropped (int): Number of dropped images.
        used (int): Number of used images.
        dx_names (list): Diagnosis names.

    Returns:
        matplotlib.figure.Figure: Class distribution plot.
    """
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot experiment counts
    sns.barplot(x=exp_counts.index, y=exp_counts.values, ax=ax, color='cyan', order=dx_names)

    bars = ax.containers[0]
    ax.bar_label(bars, padding=3, fontsize=12)

    # Overlay undersampled portion
    ax.bar(dx_names.index('nv'), dropped, bottom=exp_counts['nv'], color='gray', hatch='/', alpha=0.5)

    # Legend entries
    exp_patch = mp.Patch(color='cyan', label=f'Used Images: {used:>9}')
    dropped_patch = mp.Patch(color='gray', hatch='///', alpha=0.5, label=f'Dropped Images: {dropped}')
    ax.legend(handles=[dropped_patch, exp_patch], loc='upper left')

    ax.set_xlabel('Diagnosis')
    ax.set_ylabel('Number of Images')
    ax.set_title('Class Distribution (after undersampling)')
    ax.tick_params(axis='x', labelrotation=45)

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

    ax[0].plot(history['accuracy'], color='cyan', label='train')
    ax[0].plot(history['val_accuracy'], color='magenta', label='val')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_title(f'Accuracy ({exp_name})')
    ax[0].legend(fontsize='large', loc='lower right')
    ax[0].grid(True, linestyle='--', alpha=0.5)

    ax[1].plot(history['loss'], color='cyan', label='train')
    ax[1].plot(history['val_loss'], color='magenta', label='val')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].set_title(f'Loss ({exp_name})')
    ax[1].legend(fontsize='large', loc='upper right')
    ax[1].grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()

    return fig

def plot_cm(cm, dx_names, mode, exp_dir):
    """
    Plots a confusion matrix with normalized prediction rates.

    Args:
        cm (np.ndarray): Confusion matrix.
        dx_names (list): Diagnosis names.
        mode (str): Mode setting (val/test/ensemble).
        exp_dir (Path): Object pointing to experiment folder.

    Returns:
        matplotlib.figure.Figure: Confusion matrix plot.
    """
    exp_name = exp_dir.name

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
    ax.tick_params(axis='x', labelrotation=45)

    fig.tight_layout()

    return fig

def plot_prc(prc_data, dx_names, mode, exp_dir):
    """
    Plots a precision-recall curve overlay for each diagnosis class.

    Args:
        prc_data (dict): Precision-recall curve values for each class.
        dx_names (list): Diagnosis names.
        mode (str): Mode setting (val/test/ensemble).
        exp_dir (Path): Object pointing to experiment folder.

    Returns:
        matplotlib.figure.Figure: Precision-recall curve overlay plot.
    """
    exp_name = exp_dir.name

    cmap = matplotlib.cm.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(9, 9))

    for i, dx in enumerate(dx_names):
        precision = prc_data[dx]['precision']
        recall = prc_data[dx]['recall']
        avg_precision = prc_data[dx]['avg_precision']
        color = cmap(i)

        ax.plot(recall, precision, color=color, label=f'{dx} (AP: {avg_precision:.2f})', lw=2)

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curves ({exp_name}, {mode})')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize='large', loc='lower left')
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()

    return fig