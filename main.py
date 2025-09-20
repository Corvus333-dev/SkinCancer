import tensorflow as tf

from config import *
from scripts import export, model_ops, pipeline, plots, utils

# Experiment controller
cfg = Config(
    exp=ExpConfig(
        backbone='resnet50',
        mode='train',
        checkpoint=None,
        unfreeze=None
    ),
    train=TrainConfig(
        batch_size=32,
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
        epochs=100,
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
            model_ops.unfreeze_layers(model, cfg.exp.backbone, cfg.exp.unfreeze)
    else:
        model = model_ops.build_model(
            backbone=cfg.exp.backbone,
            input_shape=cfg.exp.input_shape,
            dropout=cfg.train.dropout
        )

    alpha = utils.calculate_class_weight(train_df, cfg.train.boost, cfg.train.focal_loss[0])
    gamma = cfg.train.focal_loss[1]
    smooth = cfg.train.focal_loss[2]

    if cfg.train.lr_decay:
        decay_steps = len(train_df) // cfg.train.batch_size * cfg.train.epochs
        warmup_steps = int(decay_steps * 0.1)
    else:
        decay_steps, warmup_steps = None, None

    model_ops.compile_model(
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

    directory = export.create_directory(cfg.exp.backbone)

    history = model_ops.train_model(
        model,
        directory,
        train_ds,
        val_ds,
        epochs=cfg.train.epochs,
        patience=cfg.train.patience
    )

    hist_plot = plots.plot_hist(history.history, directory)
    layer_state = utils.get_layer_state(model, cfg.exp.backbone)

    export.save_model(directory, model, cfg, history, hist_plot, layer_state)

def evaluate_and_predict(ds, dx_map, dx_names):
    model = tf.keras.models.load_model(cfg.exp.checkpoint)
    p, y, y_hat = model_ops.predict_dx(ds, model)

    cr, cm = utils.compute_clf_metrics(y, y_hat, dx_names)
    cm_plot = plots.plot_cm(cm, dx_names, cfg.exp.checkpoint, cfg.exp.mode)
    prc_data = utils.compute_prc(dx_names, p, y)
    prc_plot = plots.plot_prc(cfg.exp.checkpoint, cfg.exp.mode, dx_names, prc_data)

    export.save_results(dx_map, y, y_hat, cr, cm_plot, prc_data, prc_plot, cfg.exp.checkpoint, cfg.exp.mode)

def main():
    dx_map, dx_names, train_df, val_df, test_df = pipeline.load_data()

    fetch_args = dict(
        batch_size=cfg.train.batch_size,
        input_shape=cfg.exp.input_shape,
        backbone=cfg.exp.backbone
    )

    if cfg.exp.mode == 'train':
        train_ds = pipeline.fetch_dataset(train_df, **fetch_args)
        val_ds = pipeline.fetch_dataset(val_df, **fetch_args, shuffle=False)
        train(train_ds, val_ds, train_df)

    elif cfg.exp.mode == 'val':
        val_ds = pipeline.fetch_dataset(val_df, **fetch_args, shuffle=False)
        evaluate_and_predict(val_ds, dx_map, dx_names)

    elif cfg.exp.mode == 'test':
        test_ds = pipeline.fetch_dataset(test_df, **fetch_args, shuffle=False)
        evaluate_and_predict(test_ds, dx_map, dx_names)

if __name__ == '__main__':
    main()