from sklearn.metrics import classification_report, confusion_matrix

from config import ExperimentConfig
from scripts.export import *
from scripts.model import *
from scripts.plots import *
from scripts.utils import *

# Experiment controller
config = ExperimentConfig(
    architecture='efficientnetb0',
    mode='train',
    checkpoint=None,
    unfreeze=None,
    bn_freeze=True,
    boost={
        0: 1.0, # akiec
        1: 1.0, # bcc
        2: 1.0, # bkl
        3: 1.0, # df
        4: 1.0, # mel
        5: 1.0, # nv
        6: 1.0  # vasc
    },
    focal_loss=(0.5, 2.0, 0.1),
    lr_decay=True,
    input_shape=(224, 224, 3),
    batch_size=64,
    dropout=(0.5, 0.25, 0.125),
    initial_lr=1e-3,
    patience=5,
    warmup_target=None,
    weight_decay=1e-4,
    epochs=50
)

def train(train_ds, val_ds, train_df):
    if config.checkpoint:
        model = tf.keras.models.load_model(config.checkpoint)
        if config.unfreeze:
            unfreeze_layers(model, config.architecture, config.unfreeze, config.bn_freeze)
    else:
        model = build_model(architecture=config.architecture, input_shape=config.input_shape, dropout=config.dropout)

    alpha = calculate_class_weight(train_df, config.boost, config.focal_loss[0])
    gamma = config.focal_loss[1]
    smooth = config.focal_loss[2]

    if config.lr_decay:
        decay_steps = len(train_df) // config.batch_size * config.epochs
        warmup_steps = int(decay_steps * 0.1)
    else:
        decay_steps, warmup_steps = None, None

    compile_model(
        model,
        initial_lr=config.initial_lr,
        warmup_target=config.warmup_target,
        decay_steps=decay_steps,
        warmup_steps=warmup_steps,
        wd=config.weight_decay,
        alpha=alpha,
        gamma=gamma,
        smooth=smooth
    )

    layer_state = get_layer_state(model, config.architecture)
    history = train_model(model, train_ds, val_ds, epochs=config.epochs, patience=config.patience)
    directory = create_directory(config.architecture)
    hist_plot = plot_hist(history.history, directory)

    save_model(directory, model, config, layer_state, history, hist_plot)

def evaluate_and_predict(ds, dx_map, dx_names):
    model = tf.keras.models.load_model(config.checkpoint)
    p, y, y_hat = predict_dx(ds, model)

    cr = classification_report(y, y_hat, target_names=dx_names, output_dict=True)
    cm = confusion_matrix(y, y_hat)
    cm_plot = plot_cm(cm, dx_names, config)
    prc_data = compute_prc(dx_names, p, y)
    prc_plot = plot_prc(config, dx_names, prc_data)

    save_results(dx_map, y, y_hat, cr, cm_plot, prc_data, prc_plot, config)

def main():
    dx_map, dx_names, train_df, val_df, test_df = load_data()

    if config.mode == 'train':
        train_ds = fetch_dataset(
            train_df,
            batch_size=config.batch_size,
            input_shape=config.input_shape
        )
        val_ds = fetch_dataset(
            val_df,
            batch_size=config.batch_size,
            input_shape=config.input_shape,
            shuffle=False
        )
        train(train_ds, val_ds, train_df)

    elif config.mode == 'val':
        val_ds = fetch_dataset(
            val_df,
            batch_size=config.batch_size,
            input_shape=config.input_shape,
            shuffle=False
        )
        evaluate_and_predict(val_ds, dx_map, dx_names)

    elif config.mode == 'test':
        test_ds = fetch_dataset(
            test_df,
            batch_size=config.batch_size,
            input_shape=config.input_shape,
            shuffle=False
        )
        evaluate_and_predict(test_ds, dx_map, dx_names)

if __name__ == '__main__':
    main()