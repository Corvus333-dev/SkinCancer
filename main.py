from scripts.pipeline import *
from scripts.model import *

def main():
    # Dataset pipeline
    df, dx_map = encode_labels()
    df = map_image_paths(df)
    train_df, dev_df, test_df = split_data(df)
    train_ds = fetch_dataset(train_df)
    dev_ds = fetch_dataset(dev_df, shuffle=False)
    test_ds = fetch_dataset(test_df, shuffle=False)

    # Model setup and training
    model = compile_model(build_model(), lr=0.001)
    train_model(train_ds, model, epochs=10)

if __name__ == '__main__':
    main()