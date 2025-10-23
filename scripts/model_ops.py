import numpy as np
from pathlib import Path
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.applications import EfficientNetB1, ResNet50

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
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecay

from scripts.keras_objects import CBAM, FusionGate, SparseCategoricalFocalCrossentropy, SparseF1Score

def build_model(backbone, input_shape, dropout_rates, classes=7):
    """
    Instantiates a base model using EfficientNetB1 or ResNet50 architecture pretrained on ImageNet dataset, and
    attaches a custom top that includes gated metadata fusion, CBAM, dense stack, and softmax output.

    Performs the following random augmentations to input:
    brightness, contrast, horizontal/vertical flip, rotation, translation, and zoom.

    Args:
        backbone (str): Base model architecture.
        input_shape (tuple): Shape of input image.
        dropout_rates (tuple): Dropout rates applied to each dense layer (bottom to top).
        classes (int): Number of classes (i.e., skin lesion types).

    Returns:
        tf.keras.Model: Functional model with frozen base layers.
    """
    if backbone == 'efficientnetb1':
        model_type = EfficientNetB1
    elif backbone == 'resnet50':
        model_type = ResNet50
    else:
        raise AssertionError('Backbone validation should be handled by ExperimentConfig.')

    base_model = model_type(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )

    base_model.trainable = False # Recursive (freezes all sub-layers)

    # Augmentation block
    augment_layers = Sequential(
        [
            RandomBrightness(0.15),
            RandomContrast(0.15),
            RandomFlip('horizontal_and_vertical'),
            RandomRotation(0.15),
            RandomTranslation(0.15, 0.15),
            RandomZoom((-0.15, 0.15))
        ],
        name='augmentation'
    )

    # Inputs
    image_input = Input(name='image', shape=input_shape)
    meta_input = Input(name='meta', shape=(17,))

    x = augment_layers(image_input) # Explicit 'training=bool' is not required
    x = base_model(x)

    # Gated metadata fusion
    x = FusionGate()([x, meta_input])

    # Convolutional block attention module
    x = CBAM()(x)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)

    x = Dense(512, activation='swish', name='funnel_top')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rates[0])(x)

    x = Dense(256, activation='swish', name='funnel_mid')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rates[1])(x)

    x = Dense(128, activation='swish', name='funnel_bot')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rates[2])(x)

    output = Dense(classes, activation='softmax', name='softmax')(x)

    return Model(inputs=[image_input, meta_input], outputs=output, name='FuFuNet')

def unfreeze_layers(model, backbone, unfreeze):
    """
    Flags specified layers as trainable.

    Args:
        model (keras.Model): Initial convergence (top-calibrated) model.
        backbone (str): Base model architecture.
        unfreeze (int | str | tuple): Layers to unfreeze.
            - int: unfreeze from this depth upward
            - str: unfreeze from this layer name upward
            - tuple: unfreeze layers containing any of these keywords

    Returns:
        None
    """
    base_model = model.get_layer(backbone)
    base_model.trainable = True # Unfreezes parent container

    # Unfreeze from depth magnitude
    if isinstance(unfreeze, int):
        for layer in base_model.layers[-unfreeze:]:
            layer.trainable = True
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
            for layer in base_model.layers[match:]:
                layer.trainable = True
            for layer in base_model.layers[:match]:
                layer.trainable = False

    # Unfreeze layers containing keyword (rare use case)
    else:
        for layer in base_model.layers:
            layer.trainable = any(keyword in layer.name for keyword in unfreeze)

def compile_model(model, initial_lr, decay_steps, warmup_target, warmup_steps, wd, alpha, gamma, smooth):
    """
    Compiles model using AdamW optimizer and sparse categorical cross-entropy loss, with optional cosine decay learning
    rate schedule and label smoothing.

    Args:
        model (keras.Model): Model to compile.
        initial_lr (float): Starting learning rate.
        decay_steps (int): Number of steps for learning rate decay.
        warmup_target (float): Learning rate after warmup (use None for no warm-up).
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

    loss = SparseCategoricalFocalCrossentropy(alpha, gamma, smooth)
    opt = AdamW(learning_rate=lr, weight_decay=wd)
    macro_f1 = SparseF1Score(average='macro', name='macro_f1')
    model.compile(optimizer=opt, loss=loss, metrics = ['accuracy', macro_f1])

def train_model(model, train_ds, val_ds, exp_dir, epochs, patience, threshold=0.001):
    """
    Trains model on dataset using early stopping and/or best-weights checkpointing.

    Args:
        model (keras.Model): Compiled model.
        train_ds (tf.data.Dataset): Training dataset.
        val_ds (tf.data.Dataset): Validation dataset.
        exp_dir (Path): Directory to save the best checkpoint.
        epochs (int): Maximum training epochs.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        threshold (float): Minimum change in loss to qualify as an improvement.

    Returns:
        keras.callbacks.History: Training history.
    """
    filepath = exp_dir / 'model.keras'

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor='val_macro_f1',
        verbose=0,
        save_best_only=True,
        mode='max'
    )

    stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=threshold,
        patience=patience,
        verbose=1,
        restore_best_weights=False
    )

    return model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[checkpoint, stop])

def predict_dx(model, ds, dx_map):
    """
    Generates probability distributions, collects associated true diagnoses, and selects max-value predictions. Places
    this data and corresponding image IDs into a DataFrame.

    Args:
        model (keras.Model): Trained model with softmax output.
        ds (tf.data.Dataset): Batched dataset of ({meta, image, image_path}, dx_code) pairs.
        dx_map (dict): Map of diagnosis codes to diagnosis names.

    Returns:
        p (np.ndarray): Probability distributions.
        y (np.ndarray): True diagnosis indices.
        y_hat (np.ndarray): Predicted diagnosis indices.
        pred_df (pd.DataFrame): DataFrame containing prediction results (image_id, dx probability, actual, predicted)
    """
    p = model.predict(ds)
    y = np.concatenate([dx_code.numpy() for _, dx_code in ds])
    y_hat = np.argmax(p, axis=1)

    # Flatten batched dataset and decode string tensors element-wise
    image_paths = [item['image_path'].numpy().decode() for item, _ in ds.unbatch()]
    # Extract stems from decoded paths
    image_ids = [Path(p).stem for p in image_paths]

    # Place results in DataFrame
    pred_df = pd.DataFrame(p, columns=[dx_map[i] for i in range(p.shape[1])])
    pred_df['actual'] = [dx_map[i] for i in y]
    pred_df['predicted'] = [dx_map[i] for i in y_hat]
    pred_df.insert(0, 'image_id', image_ids)

    return p, y, y_hat, pred_df