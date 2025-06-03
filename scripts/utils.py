import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

def calculate_class_weights(train_df, gamma=2):
    """
    Calculates class weights using normalized exponential inverse frequency.

    Args:
        train_df (pd.DataFrame): Training split DataFrame.
        gamma (float): Exponent used to adjust weight magnitudes.

    Returns:

    """
    y = train_df['dx_code'].values
    classes, counts = np.unique(y, return_counts=True)
    num_classes = len(classes)
    num_samples = len(y)

    weights = (num_samples / (num_classes * counts))**gamma
    weights /= np.mean(weights)
    class_weight = dict(zip(classes, weights))

    return class_weight

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