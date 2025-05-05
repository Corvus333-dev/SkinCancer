import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecay

def build_resnet50(input_shape, dropout, classes = 7):
    """
    Creates base model using ResNet50 architecture pretrained on ImageNet dataset,
    then adds custom pooling, dropout, and dense layers.

    Args:
        input_shape (tuple): Shape of input image.
        dropout (float): Dropout rate.
        classes (int): Number of output classes.

    Returns:
        keras.Model: Functional model graph with attached layers and weights.
    """
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )

    base_model.trainable = False # Recursive (freezes all sub-layers)

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout)(x)
    outputs = Dense(classes, activation='softmax')(x)

    return Model(inputs, outputs)

def unfreeze_block(model, framework):
    """
    Flags last residual block as trainable.

    Args:
        model (keras.Model): Initial convergence model.
        framework (str): Base model architecture.

    Returns:
        None
    """
    base_model = model.get_layer(framework)
    base_model.trainable = True # Non-recursive (only unfreezes parent layer)
    for layer in base_model.layers:
        layer.trainable = ('conv5_block3' in layer.name)

def compile_model(model, lr, lr_decay, decay_steps, wd):
    """
    Compiles model using AdamW optimizer and sparse categorical cross-entropy loss,
    with optional learning rate schedule.

    Args:
        model (keras.Model): Model to compile.
        lr (float): Learning rate for optimizer.
        lr_decay (bool): Flag for learning rate schedule.
        decay_steps (int): Number of steps for learning rate decay.
        wd (float): Weight decay for optimizer.

    Returns:
        None
    """
    if lr_decay:
        lr = CosineDecay(initial_learning_rate=lr, decay_steps=decay_steps, alpha=0.01)

    opt = AdamW(learning_rate=lr, weight_decay=wd)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics = ['accuracy'])

def train_model(model, train_ds, class_weight, epochs, threshold=0.001):
    """
    Trains model on dataset, with early stopping callback.

    Args:
        model (keras.Model): Compiled model.
        train_ds (tf.data.Dataset): Batched and preprocessed training dataset.
        class_weight (dict): Map of diagnosis codes to associated weights.
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

    return model.fit(train_ds, class_weight=class_weight, epochs=epochs, callbacks=[stop])

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