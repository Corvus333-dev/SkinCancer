from config import ModelConfig
from scripts.pipeline import *
from scripts.model import *
from scripts.artifacts import *

config = ModelConfig(
    framework='resnet50',
    input_shape=(224, 224, 3),
    batch_size=64,
    freeze_layers=True,
    training=False,
    learning_rate=0.001,
    epochs=10
)

def main():
    # Dataset pipeline
    df, dx_map = encode_labels()
    df = map_image_paths(df)
    train_df, dev_df, test_df = split_data(df)
    train_ds = fetch_dataset(train_df)
    dev_ds = fetch_dataset(dev_df, shuffle=False)
    test_ds = fetch_dataset(test_df, shuffle=False)

    # Model setup and training
    model = build_resnet50(input_shape=config.input_shape, freeze_layers=config.freeze_layers, training=config.training)
    compile_model(model, lr=config.learning_rate)
    history = train_model(train_ds, model, epochs=config.epochs)
    metrics = model.evaluate(dev_ds, return_dict=True)

    # Save artifacts
    directory = create_directory(config.framework)
    save_artifacts(directory, model, config, history, metrics)

if __name__ == '__main__':
    main()