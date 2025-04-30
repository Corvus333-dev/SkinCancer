import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model

def build_resnet50(input_shape, classes = 7):
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
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(classes, activation='softmax')(x)

    return Model(inputs, outputs)

def unfreeze_layers(model, framework, trainable_layers):
    """
    Flags base layers as trainable, starting from the top of network.

    Args:
        model (keras.Model): Initial convergence model.
        framework (str): Base model architecture.
        trainable_layers (int): Number of layers to unfreeze.

    Returns:
        None
    """
    base_model = model.get_layer(framework)
    for layer in base_model.layers[-trainable_layers:]:
        layer.trainable = True

def compile_model(model, lr):
    """
    Compiles model using Adam optimizer and sparse categorical cross-entropy loss.

    Args:
        model (keras.Model): Model to compile.
        lr (float): Learning rate for optimizer.

    Returns:
        None
    """
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics = ['accuracy'])

def train_model(train_ds, model, epochs, threshold=0.001):
    """
    Trains model on dataset, with early stopping callback.

    Args:
        train_ds (tf.data.Dataset): Batched and preprocessed training dataset.
        model (keras.Model): Compiled model.
        epochs (int): Maximum number of epochs.
        threshold (float): Minimum change in loss to qualify as an improvement.

    Returns:
        keras.callbacks.History: Training history.
    """
    stop = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        min_delta= threshold,
        patience=3,
        verbose=1,
        restore_best_weights=True
    )

    return model.fit(train_ds, epochs=epochs, callbacks=[stop])

def predict_labels(ds, model):
    """
    Generates predicted labels and collects associated true labels.

    Args:
        ds (tf.data.Dataset): Batched dataset of (image, label) pairs.
        model (keras.Model): Trained model with softmax output.

    Returns:
        y (list): True label indices.
        y_hat (np.ndarray): Predicted label indices.
    """
    y = []
    predictions = model.predict(ds)

    for image, label in ds:
        y.extend(label.numpy())

    y_hat = np.argmax(predictions, axis=1)

    return y, y_hat