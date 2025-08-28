import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from scripts.pipeline import *

def load_data():
    df, dx_map = encode_labels()
    dx_names = list(dx_map.values())
    df = map_image_paths(df)
    df = encode_meta(df)
    train_df, val_df, test_df = split_data(df)

    return dx_map, dx_names, train_df, val_df, test_df

def calculate_class_weight(train_df, boost, gamma):
    """
    Calculates class weights using inverse frequency with adjustable exponent and class-specific multipliers.

    Args:
        train_df (pd.DataFrame): Training set DataFrame.
        boost (dict): Map of diagnosis codes and weight multipliers.
        gamma (float): Exponent used for weight magnitude.

    Returns:
        dict: Map of diagnosis codes and weights.
    """
    y = train_df['dx_code'].values
    classes, counts = np.unique(y, return_counts=True)
    weights = (len(y) / (len(classes) * counts)) ** gamma

    if boost:
        for key, value in boost.items():
            weights[key] *= value

    weights /= np.mean(weights)

    # Cast to JSON-compatible types (required for custom loss serialization)
    return {int(key): float(value) for key, value in zip(classes, weights)}

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

def compute_prc(dx_names, p, y, classes=7):
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