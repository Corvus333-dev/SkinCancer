import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.applications import EfficientNetB0, InceptionV3, ResNet50
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    RandomBrightness,
    RandomContrast,
    RandomFlip,
    RandomRotation,
    RandomTranslation,
    RandomZoom
)
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecay

from scripts.utils import SparseCategoricalFocalCrossentropy

def build_model(architecture, input_shape, dropout, classes=7):
    """
    Instantiates a base model using EfficientNetB0, InceptionV3, or ResNet50 architecture pretrained on ImageNet
    dataset, and attaches a broadly applicable custom top consisting of layers:

    GAP -> BN -> DO -> Dense-512 -> BN -> DO -> Dense-256 -> BN -> DO -> Dense-128 -> BN -> DO -> Dense-7

    Performs the following random augmentations to input:
    brightness, contrast, horizontal/vertical flip, rotation, translation, and zoom.

    Args:
        architecture (str): Base model architecture.
        input_shape (tuple): Shape of input image.
        dropout (tuple): Dropout rates (ordered from bottom to top).
        classes (int): Number of classes (i.e., skin lesion types).

    Returns:
        tf.keras.Model: Functional model with frozen base layers.
    """
    if architecture == 'efficientnetb0':
        model_type = EfficientNetB0
    elif architecture == 'inception_v3':
        model_type = InceptionV3
    elif architecture == 'resnet50':
        model_type = ResNet50
    else:
        raise AssertionError('Architecture validation should be handled by ExperimentConfig.')

    base_model = model_type(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )

    base_model.trainable = False # Recursive (freezes all sub-layers)

    augment_layers = Sequential([
        RandomBrightness(0.15),
        RandomContrast(0.15),
        RandomFlip('horizontal_and_vertical'),
        RandomRotation(0.15),
        RandomTranslation(0.15, 0.15),
        RandomZoom((-0.15, 0.15))
    ])

    inputs = Input(shape=input_shape)
    x = augment_layers(inputs) # Explicit 'training=bool' is not required
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout[0])(x)

    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout[1])(x)

    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout[2])(x)

    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout[3])(x)

    outputs = Dense(classes, activation='softmax')(x)

    return Model(inputs, outputs)

def unfreeze_layers(model, architecture, unfreeze):
    """
    Flags specified layers as trainable.

    Args:
        model (keras.Model): Initial convergence (top-calibrated) model.
        architecture (str): Base model architecture.
        unfreeze (int | str | tuple): Layer specification for unfreezing:
            - int: unfreeze from this layer depth to the top
            - str: unfreeze from this layer name to the top
            - tuple: unfreeze layers containing any of these keywords

    Returns:
        None
    """
    base_model = model.get_layer(architecture)
    base_model.trainable = True # Recursive (unfreezes all sub-layers)

    # Unfreeze from depth magnitude
    if isinstance(unfreeze, int):
        for layer in base_model.layers[:-unfreeze]:
            layer.trainable = False

    # Unfreeze from layer name
    elif isinstance(unfreeze, str):
        matches = [i for i, layer in enumerate(base_model.layers) if layer.name == unfreeze]
        if len(matches) == 0:
            raise ValueError(f'Layer named "{unfreeze}" not found.')
        elif len(matches) > 1:
            raise ValueError(f'Multiple layers named "{unfreeze}" found.')
        else:
            match = matches[0] # Single layer index needed for slicing
            for layer in base_model.layers[:match]:
                layer.trainable = False

    # Unfreeze layers containing keyword (rare use case)
    else:
        for layer in base_model.layers:
            layer.trainable = any(keyword in layer.name for keyword in unfreeze)

def compile_model(model, initial_lr, warmup_target, decay_steps, warmup_steps, wd, alpha, gamma, smooth):
    """
    Compiles model using AdamW optimizer and sparse categorical cross-entropy loss,
    with cosine decay learning rate schedule and label smoothing.

    Args:
        model (keras.Model): Model to compile.
        initial_lr (float): Starting learning rate.
        warmup_target (float): Learning rate after warmup (None for no warm-up).
        decay_steps (int): Number of steps for learning rate decay.
        warmup_steps (int): Number of steps for learning rate warmup.
        wd (float): Weight decay for optimizer.
        alpha (dict): Map of diagnosis codes (int) and weights (float). Must be JSON-compatible.
        gamma (float): Focusing parameter. Gradually reduces the importance given to easy examples.
        smooth (float): Label smoothing effect. Reduces overconfidence in predictions.

    Returns:
        None
    """
    if decay_steps:
        lr = CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            alpha=0.1,
            warmup_target=warmup_target,
            warmup_steps=warmup_steps
        )
    else:
        lr = initial_lr

    if gamma:
        loss = SparseCategoricalFocalCrossentropy(alpha, gamma, smooth)
    else:
        loss = SparseCategoricalCrossentropy()

    opt = AdamW(learning_rate=lr, weight_decay=wd)
    model.compile(optimizer=opt, loss=loss, metrics = ['accuracy'])

def train_model(model, train_ds, dev_ds, class_weight, epochs, patience, threshold=0.001):
    """
    Trains model on dataset, with early stopping callback.

    Args:
        model (keras.Model): Compiled model.
        train_ds (tf.data.Dataset): Training dataset.
        dev_ds (tf.data.Dataset): Development dataset used for monitoring validation loss.
        class_weight (dict): Map of diagnosis codes to associated weights.
        epochs (int): Maximum number of epochs.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        threshold (float): Minimum change in loss to qualify as an improvement.

    Returns:
        keras.callbacks.History: Training history.
    """
    stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=threshold,
        patience=patience,
        verbose=1,
        restore_best_weights=True
    )

    return model.fit(train_ds, validation_data=dev_ds, class_weight=class_weight, epochs=epochs, callbacks=[stop])

def predict_dx(ds, model):
    """
    Generates predicted diagnoses and collects associated true diagnoses.

    Args:
        ds (tf.data.Dataset): Batched dataset of (image, dx_code) pairs.
        model (keras.Model): Trained model with softmax output.

    Returns:
        p (np.ndarray): Probability distributions.
        y (np.ndarray): True diagnosis indices.
        y_hat (np.ndarray): Predicted diagnosis indices.
    """
    p = model.predict(ds)
    y = np.concatenate([dx_code.numpy() for _, dx_code in ds])
    y_hat = np.argmax(p, axis=1)

    return p, y, y_hat