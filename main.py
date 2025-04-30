from sklearn.metrics import classification_report, confusion_matrix

from config import ModelConfig
from scripts.model import *
from scripts.pipeline import *
from scripts.plots import *
from scripts.utils import *

config = ModelConfig(
    framework='resnet50',
    mode='train',
    checkpoint=None,
    dist_plot=False,
    input_shape=(224, 224, 3),
    batch_size=32,
    trainable_layers=0,
    training=False,
    learning_rate=0.001,
    epochs=30
)

def load_data():
    df, dx_map = encode_labels()
    df = map_image_paths(df)
    train_df, dev_df, test_df = split_data(df)

    if config.dist_plot:
        dist_plot = plot_dist(df, dx_map)
        save_dist(dist_plot)

    return dx_map, train_df, dev_df, test_df

def train(ds):
    if config.checkpoint:
        model = tf.keras.models.load_model(config.checkpoint)
        if config.trainable_layers > 0:
            unfreeze_layers(model, config.framework, config.trainable_layers)
            compile_model(model, lr=config.learning_rate)
    else:
        model = build_resnet50(
            input_shape=config.input_shape,
            training=config.training
        )
        compile_model(model, lr=config.learning_rate)

    history = train_model(ds, model, epochs=config.epochs)
    directory = create_directory(config.framework)
    hist_plot = plot_hist(history.history, directory)

    save_model(directory, model, config, history, hist_plot)

def evaluate_and_predict(ds, dx_map):
    model = tf.keras.models.load_model(config.checkpoint)
    metrics = model.evaluate(ds, return_dict=True)
    y, y_hat = predict_labels(ds, model)

    labels = list(dx_map.values())
    cr = classification_report(y, y_hat, target_names=labels, output_dict=True)
    cm = confusion_matrix(y, y_hat)
    cm_plot = plot_cm(cm, labels, config)

    save_results(dx_map, y, y_hat, cr, cm_plot, metrics, config)

def main():
    dx_map, train_df, dev_df, test_df = load_data()

    if config.mode == 'train':
        train_ds = fetch_dataset(train_df, batch_size=config.batch_size)
        train(train_ds)

    elif config.mode == 'dev':
        dev_ds = fetch_dataset(dev_df, batch_size=config.batch_size, shuffle=False)
        evaluate_and_predict(dev_ds, dx_map)

    elif config.mode == 'test':
        test_ds = fetch_dataset(test_df, batch_size=config.batch_size, shuffle=False)
        evaluate_and_predict(test_ds, dx_map)

if __name__ == '__main__':
    main()