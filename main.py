from scripts.pipeline import *
from scripts.model import *

def main():
    # Dataset pipeline
    df, dx_map = encode_labels()
    df = map_image_paths(df)
    train_df, dev_df, test_df = split_data(df)
    train_ds = fetch_dataset(train_df)

    # Model setup and training
    model = compile_model(build_model())
    train_model(train_ds, model)

if __name__ == '__main__':
    main()