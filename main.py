from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from config import ExperimentConfig
from scripts.model import *
from scripts.pipeline import *
from scripts.plots import *
from scripts.utils import *

config = ExperimentConfig(
    architecture='resnet50',
    mode='dev',
    checkpoint='models/resnet50_20250603_0323/model.keras',
    unfreeze='conv5_block1_1_conv',
    augment=True,
    class_weight=True,
    dist_plot=False,
    learning_rate_decay=True,
    input_shape=(224, 224, 3),
    batch_size=32,
    dropout=0.2,
    initial_learning_rate=1e-5,
    patience=5,
    warmup_target=1e-4,
    weight_decay=1e-6,
    epochs=50
)

def load_data():
    df, dx_map = encode_labels()
    dx_names = list(dx_map.values())
    df = map_image_paths(df)
    train_df, dev_df, test_df = split_data(df)

    if config.dist_plot:
        dist_plot = plot_dist(df, dx_names)
        save_dist(dist_plot)

    return dx_map, dx_names, train_df, dev_df, test_df

def train(ds, train_df):
    if config.checkpoint:
        model = tf.keras.models.load_model(config.checkpoint)
        if config.unfreeze:
            unfreeze_layers(model, config.architecture, config.unfreeze)
    else:
        model = build_model(architecture=config.architecture, input_shape=config.input_shape, dropout=config.dropout)

    if config.learning_rate_decay:
        decay_steps = len(train_df) // config.batch_size * config.epochs
        warmup_steps = int(decay_steps * 0.1)
    else:
        decay_steps = None
        warmup_steps = None

    compile_model(
        model,
        initial_lr=config.initial_learning_rate,
        warmup_target=config.warmup_target,
        decay_steps=decay_steps,
        warmup_steps=warmup_steps,
        wd=config.weight_decay
    )

    if config.class_weight:
        y_train = train_df['dx_code'].values
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weight = dict(zip(classes, weights))
    else:
        class_weight = None

    layer_state = get_layer_state(model, config.architecture)
    history = train_model(model, ds, class_weight, epochs=config.epochs, patience=config.patience)
    directory = create_directory(config.architecture)
    hist_plot = plot_hist(history.history, directory)

    save_model(directory, model, config, layer_state, history, hist_plot)

def evaluate_and_predict(ds, dx_map, dx_names):
    model = tf.keras.models.load_model(config.checkpoint)
    p, y, y_hat = predict_dx(ds, model)

    cr = classification_report(y, y_hat, target_names=dx_names, output_dict=True)
    cm = confusion_matrix(y, y_hat)
    cm_plot = plot_cm(cm, dx_names, config)
    prc_data = get_prc_data(dx_names, p, y)
    prc_plot = plot_prc(config, dx_names, prc_data)

    save_results(dx_map, y, y_hat, cr, cm_plot, prc_data, prc_plot, config)

def main():
    dx_map, dx_names, train_df, dev_df, test_df = load_data()

    if config.mode == 'train':
        train_ds = fetch_dataset(
            train_df,
            architecture=config.architecture,
            batch_size=config.batch_size,
            augment=config.augment
        )
        train(train_ds, train_df)

    elif config.mode == 'dev':
        dev_ds = fetch_dataset(
            dev_df,
            architecture=config.architecture,
            batch_size=config.batch_size,
            augment=False,
            shuffle=False
        )
        evaluate_and_predict(dev_ds, dx_map, dx_names)

    elif config.mode == 'test':
        test_ds = fetch_dataset(
            test_df,
            architecture=config.architecture,
            batch_size=config.batch_size,
            augment=False,
            shuffle=False
        )
        evaluate_and_predict(test_ds, dx_map, dx_names)

if __name__ == '__main__':
    main()