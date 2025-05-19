from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from config import ModelConfig
from scripts.model import *
from scripts.pipeline import *
from scripts.plots import *
from scripts.utils import *

config = ModelConfig(
    framework='resnet50',
    mode='dev',
    checkpoint='models/resnet50_20250518_1646/model.keras',
    unfreeze=('conv5_', 'conv4_block6_', 'conv4_block5_'),
    augment=True,
    class_weight=True,
    dist_plot=False,
    learning_rate_decay=True,
    input_shape=(224, 224, 3),
    batch_size=32,
    dropout=0.3,
    initial_learning_rate=0.5e-5,
    warmup_target=None,
    weight_decay=1e-5,
    epochs=25
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
            unfreeze_layers(model, config.framework, config.unfreeze)
    else:
        model = build_resnet50(input_shape=config.input_shape, dropout=config.dropout)

    if config.learning_rate_decay:
        decay_steps = len(train_df) // config.batch_size * config.epochs
        warmup_steps = int(decay_steps * 0.2)
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

    layer_state = get_layer_state(model, config.framework)
    history = train_model(model, ds, class_weight, epochs=config.epochs)
    directory = create_directory(config.framework)
    hist_plot = plot_hist(history.history, directory)

    save_model(directory, model, config, layer_state, history, hist_plot)

def evaluate_and_predict(ds, dx_map, dx_names):
    model = tf.keras.models.load_model(config.checkpoint)
    y, y_hat = predict_dx(ds, model)

    cr = classification_report(y, y_hat, target_names=dx_names, output_dict=True)
    cm = confusion_matrix(y, y_hat)
    cm_plot = plot_cm(cm, dx_names, config)

    save_results(dx_map, y, y_hat, cr, cm_plot, config)

def main():
    dx_map, dx_names, train_df, dev_df, test_df = load_data()

    if config.mode == 'train':
        train_ds = fetch_dataset(train_df, batch_size=config.batch_size, augment=config.augment)
        train(train_ds, train_df)

    elif config.mode == 'dev':
        dev_ds = fetch_dataset(dev_df, batch_size=config.batch_size, augment=False, shuffle=False)
        evaluate_and_predict(dev_ds, dx_map, dx_names)

    elif config.mode == 'test':
        test_ds = fetch_dataset(test_df, batch_size=config.batch_size, augment=False, shuffle=False)
        evaluate_and_predict(test_ds, dx_map, dx_names)

if __name__ == '__main__':
    main()