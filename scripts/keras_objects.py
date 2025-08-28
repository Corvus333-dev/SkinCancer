import tensorflow as tf
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import (
    Concatenate,
    Conv2D,
    Dense,
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
    """
    Convolutional block attention module that enhances features via channel and spatial attention mechanisms.

    Args:
            reduction_ratio (int): Reduction ratio for channel MLP bottleneck.
            kernel_size (int): Kernel size for spatial attention convolution.
            use_spatial (bool): Toggles use of spatial attention.
            name (str): Layer name.
            **kwargs: Additional keyword arguments passed to parent class.

    Reference:
            Woo, S., Park, J., Lee, J.Y. and Kweon, I.S., 2018. CBAM: Convolutional Block Attention Module.
            Proceedings of the European Conference on Computer Vision (ECCV) (pp. 3-19).
            Paper: https://arxiv.org/abs/1807.06521
    """
    def __init__(self, reduction_ratio=16, kernel_size=5, use_spatial=True, name='cbam', **kwargs):
        super().__init__(name=name, **kwargs)
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size
        self.use_spatial = use_spatial

    def build(self, input_shape):
        """
        Constructs the CBAM architecture, consisting of:
            - channel attention: two-layer MLP with reduction ratio bottleneck
            - spatial attention: single convolution processing pooled channel features
        Args:
            input_shape (tuple): Shape of input tensor (used to calculate channel size).

        Returns:
            None
        """
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
        """
        Applies channel attention, optionally followed by spatial attention to input feature maps.

        Args:
            inputs (tf.Tensor): Input feature tensor of shape (batch size, h, w, c).

        Returns:
            tf.Tensor: Feature maps with spatial attention applied.
        """
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
    Implements focal cross-entropy loss for sparse categorical classification, calculated via:

        L(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

        where:
            p_t is predicted probability
            alpha_t is class-specific weight
            gamma is focusing parameter

    Args:
            alpha (dict): Map of diagnosis codes (int) and class weights (float). Must be JSON-compatible.
            gamma (float): Focusing parameter. Gradually reduces the importance given to easy examples.
            smooth (float): Label smoothing effect. Reduces overconfidence in predictions.
            reduction (str): Type of reduction applied to loss.
            name (str): Name for the loss instance.
    """
    def __init__(self, alpha, gamma, smooth, reduction='sum_over_batch_size', name='scarface'):
        super().__init__(reduction=reduction, name=name)
        self._alpha = alpha # Serialization copy
        self.alpha = tf.constant(list(alpha.values()), dtype=tf.float32)
        self.gamma = gamma
        self.smooth = smooth

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
            'smooth': self.smooth
        })

        return config