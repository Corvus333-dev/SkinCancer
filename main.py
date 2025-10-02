from pathlib import Path
import tensorflow as tf

from config import *
from scripts import ensemble, export, model_ops, pipeline, plots, utils

# Experiment controller
cfg = Config(
    exp=ExpConfig(
        mode='train',
        backbone='resnet50',
        checkpoint=None,
        unfreeze=None,
        best_models=None
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
        focal_loss=(0.4, 1.9, 0.1),
        initial_lr=1e-3,
        lr_decay=True,
        patience=10,
        warmup_target=None,
        weight_decay=1e-4
    )
)

def train(train_ds, val_ds, train_df, exp_dir):
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

    # Focal loss parameters
    alpha = utils.calculate_class_weight(train_df, cfg.train.focal_loss[0], cfg.train.boost)
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

    history = model_ops.train_model(
        model,
        train_ds,
        val_ds,
        exp_dir,
        epochs=cfg.train.epochs,
        patience=cfg.train.patience
    )

    hist_plot = plots.plot_hist(history.history, exp_dir)
    layer_state = utils.get_layer_state(model, cfg.exp.backbone)

    export.save_model(model, cfg, history, hist_plot, layer_state, exp_dir)

def predict(ds, dx_map):
    model = tf.keras.models.load_model(cfg.exp.checkpoint)
    p, y, y_hat, pred_df = model_ops.predict_dx(model, ds, dx_map)

    return p, y, y_hat, pred_df

def evaluate(p, y, y_hat, pred_df, dx_names, exp_dir):
    cr, cm = utils.compute_clf_metrics(y, y_hat, dx_names)
    cm_plot = plots.plot_cm(cm, dx_names, cfg.exp.mode, exp_dir)
    prc_data = utils.compute_prc(p, y, dx_names)
    prc_plot = plots.plot_prc(prc_data, dx_names, cfg.exp.mode, exp_dir)

    export.save_results(pred_df, cr, cm_plot, prc_data, prc_plot, cfg.exp.mode, exp_dir)

def main():
    train_df, val_df, test_df, dx_map, dx_names = pipeline.load_data()

    fetch_args = dict(
        batch_size=cfg.train.batch_size,
        input_shape=cfg.exp.input_shape,
        backbone=cfg.exp.backbone
    )

    if cfg.exp.mode == 'train':
        exp_dir = export.make_exp_dir(cfg.exp.backbone)
        train_ds = pipeline.fetch_dataset(train_df, **fetch_args)
        val_ds = pipeline.fetch_dataset(val_df, **fetch_args, shuffle=False)
        train(train_ds, val_ds, train_df, exp_dir)

    else:
        if cfg.exp.mode == 'val':
            exp_dir = Path(cfg.exp.checkpoint).parent
            val_ds = pipeline.fetch_dataset(val_df, **fetch_args, shuffle=False)
            p, y, y_hat, pred_df = predict(val_ds, dx_map)

        elif cfg.exp.mode == 'test':
            exp_dir = Path(cfg.exp.checkpoint).parent
            test_ds = pipeline.fetch_dataset(test_df, **fetch_args, shuffle=False)
            p, y, y_hat, pred_df = predict(test_ds, dx_map)

        else: # Ensemble models
            exp_dir = export.make_exp_dir(cfg.exp.mode)
            merged_df = ensemble.merge_predictions(cfg.exp.best_models, dx_names)
            ensemble_df = ensemble.ensemble_models(merged_df, dx_names)
            p, y, y_hat, pred_df = ensemble.unpack_predictions(ensemble_df, dx_map, dx_names)

        evaluate(p, y, y_hat, pred_df, dx_names, exp_dir)

if __name__ == '__main__':
    main()