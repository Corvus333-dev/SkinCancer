import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable()
class SparseCategoricalFocalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, alpha, gamma, name='sparse_categorical_focal_crossentropy'):
        super().__init__(name=name)
        self._alpha = alpha # Serialization copy
        self.alpha = tf.constant(list(alpha.values()), dtype=tf.float32)
        self.gamma = gamma

    def call(self, y, y_hat):
        y = tf.cast(y, tf.int32)
        y_oh = tf.one_hot(y, depth=7)
        y_hat = tf.clip_by_value(y_hat, 1e-7, 1 - 1e-7)

        alpha_t = tf.reduce_sum(y_oh * self.alpha, axis=1)
        p_t = tf.reduce_sum(y_oh * y_hat, axis=1)

        return -alpha_t * tf.pow(1 - p_t, self.gamma) * tf.math.log(p_t)

    def get_config(self):
        return {
            'alpha': self._alpha,
            'gamma': self.gamma,
            'name': self.name
        }

def calculate_class_weight(train_df, gamma):
    """
    Calculates class weights using inverse frequency with adjustable exponent.

    Args:
        train_df (pd.DataFrame): Training set DataFrame.
        gamma (float): Exponent used for weight magnitude.

    Returns:
        dict: Map of diagnosis codes and weights.
    """
    y = train_df['dx_code'].values
    classes, counts = np.unique(y, return_counts=True)
    weights = (len(y) / (len(classes) * counts)) ** gamma

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