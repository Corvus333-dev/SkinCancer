import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input as ppi_efficientnet
from tensorflow.keras.applications.inception_v3 import preprocess_input as ppi_inception
from tensorflow.keras.applications.resnet import preprocess_input as ppi_resnet

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
    Prepares metadata for multimodal input by cleaning/normalizing age and one-hot encoding sex/localization. Collinear
    categories are conventionally dropped, but this is not strictly necessary for a neural network.

    Args:
        df (pd.DataFrame): DataFrame containing raw metadata.

    Returns:
        pd.DataFrame: DataFrame containing encoded metadata.
    """
    df['age'] = df['age'].fillna(0.0) / 100
    df = pd.get_dummies(df, prefix=['sex', 'loc'], columns=['sex', 'localization'], drop_first=True)

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
    Splits DataFrame into train, dev, and test sets using an 80/10/10 split with stratification. Places images from
    the same lesion within the train set to prevent data leakage and preserve natural augmentation.

    Args:
        df (pd.DataFrame): Full DataFrame, including image paths.

    Returns:
        pd.DataFrame: Train, dev, and test DataFrames.
    """

    # Temporarily remove duplicate lesions
    unique_df = df.drop_duplicates(subset=['lesion_id'])

    # Split so that duplicate lesions are coupled
    train_unique_df, temp_unique_df = train_test_split(
        unique_df,
        test_size=0.2,
        stratify=unique_df['dx_code'],
        random_state=9
    )
    dev_unique_df, test_unique_df = train_test_split(
        temp_unique_df,
        test_size=0.5,
        stratify=temp_unique_df['dx_code'],
        random_state=9
    )

    # Reintroduce duplicate lesions
    train_df = df[df['lesion_id'].isin(train_unique_df['lesion_id'])]

    return train_df, dev_unique_df, test_unique_df

def preprocess_image(path, architecture):
    """
    Decodes a JPEG-encoded image, resizes with pad, and performs ImageNet-style normalization.

    Args:
        path (tf.Tensor): Image path.
        dx_code (tf.Tensor): Encoded label associated with its image path.
        architecture (str): Base model architecture.

    Returns:
        image (tf.Tensor): Preprocessed image of shape (224, 224, 3) or (299, 299, 3).
        dx_code (tf.Tensor): Corresponding input image diagnosis code.
    """
    if architecture == 'efficientnetb0':
        preprocess_input = ppi_efficientnet
        th, tw = 224, 224
    elif architecture == 'inception_v3':
        preprocess_input = ppi_inception
        th, tw = 299, 299
    elif architecture == 'resnet50':
        preprocess_input = ppi_resnet
        th, tw = 224, 224
    else:
        raise AssertionError('Architecture validation should be handled by ExperimentConfig.')

    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_with_pad(image, target_height=th, target_width=tw)

    image = preprocess_input(image)

    return image

def fetch_dataset(df, architecture, batch_size, shuffle=True):
    """
    Creates a Tensorflow Dataset from corresponding metadata, preprocessed images, and diagnosis codes.

    Args:
        df (pd.DataFrame): DataFrame with image paths and diagnosis codes.
        architecture (str): Base model architecture.
        batch_size (int): Samples per batch.
        shuffle (bool): Optional shuffling.

    Returns:
        tf.data.Dataset: Batched dataset of (image, dx_code) pairs.
    """
    meta_columns = [col for col in df.columns if col == 'age' or col.startswith('sex_') or col.startswith('loc_')]
    meta_array = df[meta_columns].to_numpy().astype('float32')
    paths = df['image_path'].values
    dx_codes = df['dx_code'].values

    ds = tf.data.Dataset.from_tensor_slices((meta_array, paths, dx_codes))

    def preprocess_all(meta_vector, path, dx_code):
        # Package metadata, preprocessed image, and dx_code
        image = preprocess_image(path, architecture)
        return {'meta': meta_vector, 'image': image}, dx_code

    ds = ds.map(preprocess_all, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=2000)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds