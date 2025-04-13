import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model

def build_model(input_shape = (224, 224, 3), classes = 7):
    """
    Creates base model using ResNet50 architecture pretrained on ImageNet dataset,
    then adds custom pooling, dropout, and dense layers with softmax activation.

    Args:
        input_shape (tuple): Shape of input image.
        classes (int): Number of output classes.

    Returns:
        keras.Model: Functional model graph with attached layers and weights.
    """
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )

    base_model.trainable = False

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(classes, activation='softmax')(x)

    return Model(inputs, outputs)

def compile_model(model, lr=0.001):
    """
    Compiles model using Adam optimizer and sparse categorical cross-entropy loss.

    Args:
        model (keras.Model): Model to compile.
        lr (float): Learning rate for optimizer.

    Returns:
        keras.Model: Compiled model.
    """
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def train_model(train_ds, model, epochs=10, threshold=0.001):
    """
    Trains model on dataset, with early stopping callback.

    Args:
        train_ds (tf.data.Dataset): Batched and preprocessed training dataset.
        model (keras.Model): Compiled model.
        epochs (int): Maximum number of epochs.
        threshold (float): Minimum change in loss to qualify as an improvement.

    Returns:
        History: Keras object containing training metrics.
    """
    stop = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        min_delta= threshold,
        patience=3,
        verbose=1,
        restore_best_weights=True
    )

    return model.fit(train_ds, epochs=epochs, callbacks=[stop])