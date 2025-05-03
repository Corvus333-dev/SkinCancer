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
    checkpoint='models/resnet50_20250502_1639/model.keras',
    augment=True,
    class_weight=True,
    dist_plot=False,
    freeze=False,
    input_shape=(224, 224, 3),
    batch_size=32,
    learning_rate=1e-6,
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
        if not config.freeze:
            unfreeze_block(model, config.framework)
            compile_model(model, lr=config.learning_rate)
    else:
        model = build_resnet50(input_shape=config.input_shape)
        compile_model(model, lr=config.learning_rate)

    if config.class_weight:
        y_train = train_df['dx_code'].values
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weight = dict(zip(classes, weights))
    else:
        class_weight = None

    history = train_model(model, ds, class_weight, epochs=config.epochs)
    directory = create_directory(config.framework)
    hist_plot = plot_hist(history.history, directory)

    save_model(directory, model, config, history, hist_plot)

def evaluate_and_predict(ds, dx_map, dx_names):
    model = tf.keras.models.load_model(config.checkpoint)
    metrics = model.evaluate(ds, return_dict=True)
    y, y_hat = predict_labels(ds, model)

    cr = classification_report(y, y_hat, target_names=dx_names, output_dict=True)
    cm = confusion_matrix(y, y_hat)
    cm_plot = plot_cm(cm, dx_names, config)

    save_results(dx_map, y, y_hat, cr, cm_plot, metrics, config)

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