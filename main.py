from config import ModelConfig
from scripts.pipeline import *
from scripts.model import *
from scripts.artifacts import *

config = ModelConfig(
    framework='resnet50',
    mode='train',
    checkpoint=None,
    input_shape=(224, 224, 3),
    batch_size=64,
    freeze_layers=True,
    training=False,
    learning_rate=0.001,
    epochs=20
)

def load_data():
    df, dx_map = encode_labels()
    df = map_image_paths(df)
    train_df, dev_df, test_df = split_data(df)

    return dx_map, train_df, dev_df, test_df

def train_and_save(train_df, dev_df):
    train_ds = fetch_dataset(train_df, batch_size=config.batch_size)
    dev_ds = fetch_dataset(dev_df, batch_size=config.batch_size, shuffle=False)

    if config.checkpoint:
        model = tf.keras.models.load_model(config.checkpoint)
    else:
        model = build_resnet50(
            input_shape=config.input_shape,
            freeze_layers=config.freeze_layers,
            training=config.training
        )
        compile_model(model, lr=config.learning_rate)

    history = train_model(train_ds, model, epochs=config.epochs)
    metrics = model.evaluate(dev_ds, return_dict=True)

    directory = create_directory(config.framework)
    save_training_artifacts(directory, model, config, history, metrics)

def predict_and_save(ds, dx_map):
    model = tf.keras.models.load_model(config.checkpoint)
    y, y_hat = predict_labels(ds, model)
    save_prediction_artifacts(dx_map, y, y_hat, config)

def main():
    dx_map, train_df, dev_df, test_df = load_data()

    if config.mode == 'train':
        train_and_save(train_df, dev_df)

    elif config.mode == 'validate':
        dev_ds = fetch_dataset(dev_df, batch_size=config.batch_size, shuffle=False)
        predict_and_save(dev_ds, dx_map)

    elif config.mode == 'test':
        test_ds = fetch_dataset(test_df, batch_size=config.batch_size, shuffle=False)
        predict_and_save(test_ds, dx_map)

if __name__ == '__main__':
    main()