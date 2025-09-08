import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

def encode_labels():
    """
    Assigns a unique number to each type of skin lesion diagnosis.

    Returns:
        pd.DataFrame: DataFrame containing diagnosis codes and metadata.
        dict: Map of diagnosis codes to be used as labels.
    """
    data_path = 'data/HAM10000_metadata'
    df = pd.read_csv(data_path)

    # Map diagnosis names to numerical values
    le = LabelEncoder()
    df['dx_code'] = le.fit_transform(df['dx'])
    dx_map = dict(enumerate(le.classes_))

    return df, dx_map

def encode_meta(df):
    """
    Prepares metadata for multimodal input by cleaning/normalizing age and one-hot encoding diagnosis type, sex,
    localization, and dataset origin. Collinear categories are conventionally dropped.

    Args:
        df (pd.DataFrame): DataFrame containing raw metadata.

    Returns:
        pd.DataFrame: DataFrame containing encoded metadata.
    """
    df['age'] = df['age'].fillna(0.0) / 100 # Sentinel value of 0.0 for missing ages
    df = pd.get_dummies(
        df,
        prefix=['dx_type', 'sex', 'localization', 'dataset'],
        columns=['dx_type', 'sex', 'localization', 'dataset'],
        drop_first=True
    )

    return df

def map_image_paths(df):
    """
    Joins image file names to corresponding directory paths, then adds full path to DataFrame.

    Args:
        df (pd.DataFrame): Dataframe created during label encoding.

    Returns:
        pd.DataFrame: Updated DataFrame containing image paths.
    """
    image_folders = [Path('data/HAM10000_images_part_1/'), Path('data/HAM10000_images_part_2/')]
    image_map = {}
    image_paths = []

    # Join folder path with file name
    for folder_path in image_folders:
        image_paths.extend(list(folder_path.glob('*.jpg')))

    # Map image id to image path
    for image_path in image_paths:
        image_id = image_path.stem
        image_map[image_id] = str(image_path)

    df['image_path'] = df['image_id'].map(image_map)

    return df

def split_data(df):
    """
    Splits DataFrame into training, validation, and test sets using an initial 75/15/10 split with stratification after
    dropping duplicate lesion ids. Places any images from the same lesion ID within the training set to prevent data
    leakage and provide natural augmentation (this results in a final split of ~79/13/8 for HAM10000 dataset).

    Args:
        df (pd.DataFrame): Full DataFrame, including image paths.

    Returns:
        pd.DataFrame: Training, validation, and test DataFrames.
    """

    # Temporarily remove duplicate lesions
    unique_df = df.drop_duplicates(subset=['lesion_id'])

    train_unique_df, temp_unique_df = train_test_split(
        unique_df,
        test_size=0.25,
        stratify=unique_df['dx_code'],
        random_state=9
    )
    val_unique_df, test_unique_df = train_test_split(
        temp_unique_df,
        test_size=0.4,
        stratify=temp_unique_df['dx_code'],
        random_state=9
    )

    # Reintroduce duplicate minority lesions to training set
    train_df = df[(df['lesion_id'].isin(train_unique_df['lesion_id'])) & (df['dx'] != 5)]

    return train_df, val_unique_df, test_unique_df

def preprocess_image(path, input_shape):
    """
    Decodes a JPEG-encoded image and resizes with pad.

    Note: Additional preprocessing is included in the model using a Rescaling layer for EfficientNet.

    Args:
        path (tf.Tensor): Image path.
        input_shape (tuple): Image shape (h, w, c) for model input.

    Returns:
        image (tf.Tensor): Resized and padded image of shape (h, w, c).
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=input_shape[2])
    image = tf.image.resize_with_pad(image, target_height=input_shape[0], target_width=input_shape[1])

    return image

def fetch_dataset(df, batch_size, input_shape, shuffle=True):
    """
    Creates a Tensorflow Dataset from corresponding metadata, preprocessed images, and diagnosis codes.

    Args:
        df (pd.DataFrame): DataFrame with image paths and diagnosis codes.
        batch_size (int): Samples per batch.
        input_shape (tuple): Image shape (h, w, c) for model input.
        shuffle (bool): Optional shuffling.

    Returns:
        tf.data.Dataset: Batched dataset of (meta, image, dx_code) triplets.
    """
    meta_prefixes = ['dx_type_', 'sex_', 'localization_', 'dataset_']
    meta_columns = [col for col in df.columns if col == 'age' or any(col.startswith(p) for p in meta_prefixes)]
    meta_array = df[meta_columns].to_numpy().astype('float32')
    paths = df['image_path'].values
    dx_codes = df['dx_code'].values

    ds = tf.data.Dataset.from_tensor_slices((meta_array, paths, dx_codes))

    def preprocess_all(meta_vector, path, dx_code):
        # Package metadata, preprocessed image, and dx_code
        image = preprocess_image(path, input_shape)
        return {'meta': meta_vector, 'image': image}, dx_code

    ds = ds.map(preprocess_all, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=2000)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds