import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
import tensorflow as tf
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import (
    Concatenate,
    Conv2D, Dense,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Layer,
    Multiply,
    Reshape
)
from tensorflow.keras.losses import Loss
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable()
class CBAM(Layer):
    def __init__(self, reduction_ratio=16, kernel_size=7, use_spatial=True, name='cbam', **kwargs):
        super().__init__(name=name, **kwargs)
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        self.use_spatial = use_spatial

    def build(self, input_shape):
        self.channels = input_shape[-1]

        self.mlp_dense1 = Dense(
            self.channels // self.reduction_ratio,
            activation='swish',
            kernel_initializer='he_normal',
            use_bias=False,
            name='channel_mlp_1'
        )
        self.mlp_dense2 = Dense(
            self.channels,
            kernel_initializer='he_normal',
            use_bias=False,
            name='channel_mlp_2'
        )
        self.spatial_conv = Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            strides=1,
            padding='same',
            activation='sigmoid',
            use_bias=False,
            name='spatial_conv'
        )

        super().build(input_shape)

    def channel_attention(self, inputs):
        avg_pool = GlobalAveragePooling2D()(inputs)
        max_pool = GlobalMaxPooling2D()(inputs)

        mlp_avg = self.mlp_dense2(self.mlp_dense1(avg_pool))
        mlp_max = self.mlp_dense2(self.mlp_dense1(max_pool))

        channel_att = sigmoid(mlp_avg + mlp_max)
        channel_att = Reshape((1, 1, self.channels))(channel_att)

        return Multiply()([inputs, channel_att])

    def spatial_attention(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = Concatenate(axis=-1)([avg_pool, max_pool])
        spatial_att = self.spatial_conv(concat)

        return Multiply()([inputs, spatial_att])

    def call(self, inputs):
        x = self.channel_attention(inputs)

        if self.use_spatial:
            x = self.spatial_attention(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'reduction_ratio': self.reduction_ratio,
            'kernel_size': self.kernel_size,
            'use_spatial': self.use_spatial
        })

        return config

@register_keras_serializable()
class SparseCategoricalFocalCrossentropy(Loss):
    """
    Custom focal loss implementation for sparse categorical classification, calculated via:

        L(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

        where:
            p_t is predicted probability
            alpha_t is class-specific weight
            gamma is focusing parameter
    """
    def __init__(self, alpha, gamma, smooth, reduction='sum_over_batch_size', name='sparse_categorical_focal_crossentropy'):
        """
        Initializes sparse categorical focal crossentropy loss with cosine similarity.

        Args:
            alpha (dict): Map of diagnosis codes (int) and weights (float). Must be JSON-compatible.
            gamma (float): Focusing parameter. Gradually reduces the importance given to easy examples.
            smooth (float): Label smoothing effect. Reduces overconfidence in predictions.
            name: Optional name for the loss instance.
        """
        super().__init__(reduction=reduction, name=name)
        self._alpha = alpha # Serialization copy
        self.alpha = tf.constant(list(alpha.values()), dtype=tf.float32)
        self.gamma = gamma
        self.smooth = smooth
        self.cos_lambda = cos_lambda

    def call(self, y, y_hat):
        """
        Calculates focal loss between true and predicted classes.

        Args:
            y (tf.Tensor): True diagnosis indices.
            y_hat (tf.Tensor): Probability distributions.

        Returns:
            tf.Tensor: Focal loss values.
        """
        y = tf.cast(y, tf.int32)
        y_oh = tf.one_hot(y, depth=7)
        y_sm = y_oh * (1 - self.smooth) + self.smooth / 7 # Label smoothing
        y_hat = tf.clip_by_value(y_hat, 1e-7, 1 - 1e-7) # Prevents numerical instability

        alpha_t = tf.reduce_sum(y_oh * self.alpha, axis=1)
        p_t = tf.reduce_sum(y_sm * y_hat, axis=1)
        focal_loss = -alpha_t * tf.pow(1 - p_t, self.gamma) * tf.math.log(p_t)

        return focal_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self._alpha,
            'gamma': self.gamma,
            'smooth': self.smooth,
            'cos_lambda': self.cos_lambda
        })

        return config

def calculate_class_weight(train_df, boost, gamma):
    """
    Calculates class weights using inverse frequency with adjustable exponent and (optional) class-specific multipliers.

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