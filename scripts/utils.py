import numpy as np
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, precision_recall_curve

def calculate_class_weight(train_df, gamma, boost):
    """
    Calculates class weights using inverse frequency with adjustable exponent and class-specific multipliers.

    Args:
        train_df (pd.DataFrame): Training set DataFrame.
        gamma (float): Exponent used for weight magnitude.
        boost (dict): Map of diagnosis codes and weight multipliers.

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

def get_layer_state(model, backbone):
    """
    Extracts layer names and associated training states (unfrozen or frozen) from the base model.

    Args:
        model (keras.Model): Base model object.
        backbone (str): Base model architecture.

    Returns:
        dict: Map of layer names and training states.
    """
    base_model = model.get_layer(backbone)
    layer_state = {}

    for layer in base_model.layers:
        state = 'unfrozen' if layer.trainable else 'frozen'
        layer_state[layer.name] = state

    return layer_state

def compute_clf_metrics(y, y_hat, dx_names):
    """
    Computes classification report and confusion matrix.

    Args:
        y (np.ndarray): True diagnosis indices.
        y_hat (np.ndarray): Predicted diagnosis indices.
        dx_names (list): Diagnosis names.

    Returns:
        cr (dict): Per-class metrics and aggregate averages (precision, recall, F1, support).
        cm (np.ndarray): Confusion matrix of raw counts.
    """
    cr = classification_report(y, y_hat, target_names=dx_names, output_dict=True)
    cm = confusion_matrix(y, y_hat)

    return cr, cm

def compute_prc(p, y, dx_names):
    """
    Computes precision, recall, thresholds, and average precision values for each diagnosis class.

    Args:
        p (np.ndarray): Probability distributions.
        y (np.ndarray): True diagnosis indices.
        dx_names (list): Diagnosis names.

    Returns:
        dict: Precision-recall curve values for each class.
    """
    prc_data = {}

    for i in range(len(dx_names)):
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