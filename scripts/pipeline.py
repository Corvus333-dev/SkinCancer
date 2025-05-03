import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

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
    Splits DataFrame into train, dev, and test sets using a 70/15/15 split with stratification.

    Args:
        df (pd.DataFrame): Full DataFrame, including image paths.

    Returns:
        pd.DataFrame: Train, dev, and test DataFrames.
    """
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['dx_code'], random_state=9)
    dev_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['dx_code'], random_state=9)

    return train_df, dev_df, test_df

def augment_image(image):
    """
    Performs random augmentation of images via horizontal and/or vertical flipping.

    Args:
        image (tf.Tensor): Decoded and resized image.

    Returns:
        image (tf.Tensor): Augmented or un-augmented image.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    return image

def preprocess_image(path, dx_code, augment):
    """
    Decodes a JPEG-encoded image, resizes and pads to 224x224, and performs ImageNet-style normalization.

    Args:
        path (tf.Tensor): Image path.
        dx_code (tf.Tensor): Encoded label associated with image path.
        augment (bool): Augmentation flag.

    Returns:
        image (tf.Tensor): Preprocessed image of shape (224, 224, 3).
        dx_code (tf.Tensor): Corresponding input image diagnosis code.
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_with_pad(image, target_height=224, target_width=224)

    if augment:
        image = augment_image(image)

    image = preprocess_input(image)

    return image, dx_code

def fetch_dataset(df, batch_size, augment, shuffle=True):
    """
    Creates a Tensorflow Dataset from preprocessed images and corresponding diagnosis codes.

    Args:
        df (pd.DataFrame): DataFrame with image paths and diagnosis codes.
        batch_size (int): Samples per batch.
        augment (bool): Augmentation flag.
        shuffle (bool): Optional shuffling.

    Returns:
        tf.data.Dataset: Batched dataset of (image, dx_code) pairs.
    """
    paths = df['image_path'].values
    dx_codes = df['dx_code'].values

    ds = tf.data.Dataset.from_tensor_slices((paths, dx_codes))
    ds = ds.map(lambda x, y: preprocess_image(x, y, augment), num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=1000)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds