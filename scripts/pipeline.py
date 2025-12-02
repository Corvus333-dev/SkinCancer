import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input as ppd
from tensorflow.keras.applications.resnet import preprocess_input as ppr

# Create project root path relative to this module
ROOT = Path(__file__).resolve().parent.parent

def encode_labels(df):
    """
    Assigns a unique number (diagnosis code) to each type of skin lesion (diagnosis name).

    Returns:
        df (pd.DataFrame): DataFrame containing diagnosis codes and metadata.
        dx_map (dict): Map of diagnosis codes to diagnosis names.
    """
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
        prefix=['sex', 'localization'],
        columns=['sex', 'localization'],
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
    image_folders = [ROOT / 'data/HAM10000_images_part_1', ROOT / 'data/HAM10000_images_part_2']
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

def split_data(df, seed):
    """
    Splits DataFrame into training, validation, and test sets using an initial 72/17/11 ratio. Lesion groups with
    multiple images are removed to prevent data leakage, label-based stratification is applied, then minority lesion
    groups with multiple images are added to the training set to address class imbalance, leverage natural augmentation
    from varied clinical settings, and complement artificial augmentation.

    Notes:
        - This results in an effective split of approximately 80/12/8 for HAM10000 dataset.
        - The effect of bias towards single-image majority samples has been considered.
        - Assumes dx_code 5 represents majority class (nv)

    Args:
        df (pd.DataFrame): Full DataFrame, including image paths.
        seed (int): Controls random number generator used to split dataset.

    Returns:
        pd.DataFrame: Training, validation, and test DataFrames (train_df, val_unique_df, test_unique_df).
    """

    dup_ids = df['lesion_id'][df['lesion_id'].duplicated()].unique()
    unique_df = df.drop_duplicates('lesion_id')
    unique_df = unique_df[~unique_df['lesion_id'].isin(dup_ids)]

    train_unique_df, temp_unique_df = train_test_split(
        unique_df,
        test_size=0.28,
        stratify=unique_df['dx_code'],
        random_state=seed
    )
    val_unique_df, test_unique_df = train_test_split(
        temp_unique_df,
        test_size=0.4,
        stratify=temp_unique_df['dx_code'],
        random_state=seed
    )

    # Reintroduce all duplicate minority lesions to training set
    minority_dup_df = df[df['lesion_id'].isin(dup_ids) & (df['dx_code'] != 5)]
    train_df = pd.concat([train_unique_df, minority_dup_df], ignore_index=True)

    return train_df, val_unique_df, test_unique_df

def load_data(seed):
    """
    Executes initial pipeline, converting raw data into prepared datasets.

    Args:
        seed (int): Seed for random number generator.

    Returns:
        pd.DataFrame: Training, validation, and test DataFrames (train_df, val_df, test_df).
        dx_map (dict): Map of diagnosis codes to diagnosis names.
        dx_names (list): Diagnosis names.
    """
    df = pd.read_csv(ROOT / 'data/HAM10000_metadata')
    df, dx_map = encode_labels(df)
    dx_names = list(dx_map.values())
    df = map_image_paths(df)
    df = encode_meta(df)
    train_df, val_df, test_df = split_data(df, seed)

    return train_df, val_df, test_df, dx_map, dx_names

def preprocess_image(path, input_shape, backbone):
    """
    Decodes a JPEG-encoded image and resizes with pad.

    Note: EfficientNet normalization to [0, 1] is included in the model via a Rescaling layer.

    Args:
        path (tf.Tensor): Image path.
        input_shape (tuple): Image shape (h, w, c) for model input.
        backbone (str): Base model architecture.

    Returns:
        image (tf.Tensor): Resized and padded image of shape (h, w, c).
    """
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=input_shape[2])
    image = tf.image.resize_with_pad(image, target_height=input_shape[0], target_width=input_shape[1])

    # Preprocess with respect to ImageNet
    if backbone == 'densenet121':
        image = ppd(image) # Rescale to 0-1 range and standardize
    elif backbone == 'resnet50':
        image = ppr(image) # Convert RGB to BGR and normalize

    return image

def fetch_dataset(df, batch_size, input_shape, backbone, shuffle=True):
    """
    Creates a Tensorflow Dataset from corresponding metadata, preprocessed images, and diagnosis codes.

    Args:
        df (pd.DataFrame): DataFrame with image paths and diagnosis codes.
        batch_size (int): Samples per batch.
        input_shape (tuple): Image shape (h, w, c) for model input.
        backbone (str): Base model architecture.
        shuffle (bool): Optional shuffling.

    Returns:
        tf.data.Dataset: Batched dataset of ({meta, image, image_path}, dx_code) pairs.
        n_classes (int): Number of classes.
        n_meta_features (int): Number of metadata features.
    """
    meta_prefixes = ['dx_type_', 'sex_', 'localization_', 'dataset_']
    meta_columns = [col for col in df.columns if col == 'age' or any(col.startswith(p) for p in meta_prefixes)]
    meta_array = df[meta_columns].to_numpy().astype('float32')
    paths = df['image_path'].values
    dx_codes = df['dx_code'].values

    # Dynamically count classes and metadata features for flexible dataset support
    n_classes = df['dx_code'].nunique()
    n_meta_features = meta_array.shape[1]

    ds = tf.data.Dataset.from_tensor_slices((meta_array, paths, dx_codes))

    # Package metadata, preprocessed image, image paths (used for ID tracking), and dx_code
    def preprocess_all(meta_vector, path, dx_code):
        image = preprocess_image(path, input_shape, backbone)
        return {'meta': meta_vector, 'image': image, 'image_path': path}, dx_code

    ds = ds.map(preprocess_all, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=2000)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds, n_classes, n_meta_features