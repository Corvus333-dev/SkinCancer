from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

from config import Config, ExpConfig, TrainConfig
from scripts.export import create_directory, save_model, save_results
from scripts.model import build_model, compile_model, predict_dx, train_model, unfreeze_layers
from scripts.pipeline import load_data, fetch_dataset
from scripts.plots import plot_cm, plot_hist, plot_prc
from scripts.utils import calculate_class_weight, compute_prc, get_layer_state

cfg = Config(
    exp=ExpConfig(
        architecture='resnet50v2',
        mode='train',
        checkpoint=None,
        freeze_bn=False,
        unfreeze=None
    ),
    train=TrainConfig(
        batch_size=64,
        boost={
            0: 1.0,  # akiec
            1: 1.0,  # bcc
            2: 1.0,  # bkl
            3: 1.0,  # df
            4: 1.0,  # mel
            5: 1.0,  # nv
            6: 1.0   # vasc
        },
        dropout=(0.5, 0.25, 0.125),
        epochs=1,
        focal_loss=(0.5, 2.0, 0.1),
        initial_lr=1e-3,
        lr_decay=True,
        patience=10,
        warmup_target=None,
        weight_decay=1e-4
    )
)

def train(train_ds, val_ds, train_df):
    if cfg.exp.checkpoint:
        model = tf.keras.models.load_model(cfg.exp.checkpoint)
        if cfg.exp.unfreeze:
            unfreeze_layers(model, cfg.exp.architecture, cfg.exp.unfreeze, cfg.exp.freeze_bn)
    else:
        model = build_model(
            architecture=cfg.exp.architecture,
            input_shape=cfg.exp.input_shape,
            dropout=cfg.train.dropout
        )

    alpha = calculate_class_weight(train_df, cfg.train.boost, cfg.train.focal_loss[0])
    gamma = cfg.train.focal_loss[1]
    smooth = cfg.train.focal_loss[2]

    if cfg.train.lr_decay:
        decay_steps = len(train_df) // cfg.train.batch_size * cfg.train.epochs
        warmup_steps = int(decay_steps * 0.1)
    else:
        decay_steps, warmup_steps = None, None

    compile_model(
        model,
        initial_lr=cfg.train.initial_lr,
        decay_steps=decay_steps,
        warmup_target=cfg.train.warmup_target,
        warmup_steps=warmup_steps,
        wd=cfg.train.weight_decay,
        alpha=alpha,
        gamma=gamma,
        smooth=smooth
    )

    directory = create_directory(cfg.exp.architecture)
    history = train_model(model, directory, train_ds, val_ds, epochs=cfg.train.epochs, patience=cfg.train.patience)
    hist_plot = plot_hist(history.history, directory)
    layer_state = get_layer_state(model, cfg.exp.architecture)

    save_model(directory, model, cfg, history, hist_plot, layer_state)

def evaluate_and_predict(ds, dx_map, dx_names):
    model = tf.keras.models.load_model(cfg.exp.checkpoint)
    p, y, y_hat = predict_dx(ds, model)

    cr = classification_report(y, y_hat, target_names=dx_names, output_dict=True)
    cm = confusion_matrix(y, y_hat)
    cm_plot = plot_cm(cm, dx_names, cfg.exp.checkpoint, cfg.exp.mode)
    prc_data = compute_prc(dx_names, p, y)
    prc_plot = plot_prc(cfg.exp.checkpoint, cfg.exp.mode, dx_names, prc_data)

    save_results(dx_map, y, y_hat, cr, cm_plot, prc_data, prc_plot, cfg.exp.checkpoint, cfg.exp.mode)

def main():
    dx_map, dx_names, train_df, val_df, test_df = load_data()
    batch_size = cfg.train.batch_size
    input_shape = cfg.exp.input_shape
    architecture = cfg.exp.architecture

    if cfg.exp.mode == 'train':
        train_ds = fetch_dataset(
            train_df,
            batch_size=batch_size,
            input_shape=input_shape,
            architecture=architecture
        )
        val_ds = fetch_dataset(
            val_df,
            batch_size=batch_size,
            input_shape=input_shape,
            architecture=architecture,
            shuffle=False
        )
        train(train_ds, val_ds, train_df)

    elif cfg.exp.mode == 'val':
        val_ds = fetch_dataset(
            val_df,
            batch_size=batch_size,
            input_shape=input_shape,
            architecture=architecture,
            shuffle=False
        )
        evaluate_and_predict(val_ds, dx_map, dx_names)

    elif cfg.exp.mode == 'test':
        test_ds = fetch_dataset(
            test_df,
            batch_size=batch_size,
            input_shape=input_shape,
            architecture=architecture,
            shuffle=False
        )
        evaluate_and_predict(test_ds, dx_map, dx_names)

if __name__ == '__main__':
    main()