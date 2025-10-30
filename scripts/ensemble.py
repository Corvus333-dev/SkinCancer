import pandas as pd
from pathlib import Path

def merge_predictions(model_pool, dx_names):
    """
    Merges per-model prediction DataFrames on 'image_id' into a single DataFrame. Class probability columns are renamed
    with a timestamp suffix (to distinguish models). Model-specific 'predicted' columns are dropped. One shared 'actual'
    (ground truth diagnosis) column is kept.

    Args:
         model_pool (tuple): Paths to models selected for ensembling (minimum of two required).
         dx_names (list): Diagnosis names.

    Returns:
        pd.DataFrame: Merged DataFrame containing, for each sample:
            - image_id
            - per-model probability distributions (with suffixed column names)
            - ground truth diagnosis
    """
    dfs = []

    # Load and format DataFrames
    for model in model_pool:
        exp_dir = Path(model).parent
        timestamp = exp_dir.name
        df = pd.read_csv(exp_dir / 'val_predictions.csv')
        df = df.drop(columns='predicted')
        df = df.rename(columns={dx_name: f'{dx_name}_{timestamp}' for dx_name in dx_names})
        dfs.append(df)

    left_df = dfs[0]

    # Merge DataFrames
    for df in dfs[1:]:
        df = df.drop(columns='actual')
        left_df = left_df.merge(df, on='image_id')

    return left_df

def ensemble_models(merged_df, dx_names):
    """
    Computes ensemble predictions by averaging class probabilities across models. For each sample, mean probabilities
    are calculated per diagnosis. The predicted class is assigned as the diagnosis with the maximum mean probability.
    Results are combined with image IDs and ground truths.

    Args:
         merged_df (pd.DataFrame): Merged DataFrame containing per-model probability distributions and ground truths.
         dx_names (list): Diagnosis names.

    Returns:
        pd.DataFrame: Ensemble DataFrame containing:
            - image_id
            - mean class probability distribution (one column per dx_name)
            - actual (ground truth diagnosis name)
            - predicted (ensemble-predicted diagnosis name)
    """
    prob_cols = [c for c in merged_df.columns if any(c.startswith(dx_name) for dx_name in dx_names)]
    mean_probs = merged_df[prob_cols].T.groupby(lambda c: c.split('_')[0]).mean().T
    mean_probs['predicted'] = mean_probs.idxmax(axis=1)

    ensemble_df = pd.concat([merged_df[['image_id', 'actual']], mean_probs], axis=1)
    cols = ['image_id'] + dx_names + ['actual', 'predicted']
    ensemble_df = ensemble_df[cols]

    return ensemble_df

def unpack_predictions(ensemble_df, dx_map, dx_names):
    """
    Extracts NumPy arrays for ensemble evaluation. Retrieves mean probability distributions, ground truths, and
    predictions. Ground truth and predicted diagnosis names are reverse-mapped to sparse categorical labels.

    Args:
        ensemble_df (pd.DataFrame): Ensemble DataFrame.
        dx_map (dict): Map of diagnosis codes to diagnosis names.
        dx_names (list): Diagnosis names.

    Returns:
        p (np.ndarray): Probability distributions.
        y (np.ndarray): True diagnosis indices.
        y_hat (np.ndarray): Predicted diagnosis indices.
        ensemble_df (pd.DataFrame): Ensemble DataFrame (for export).
    """
    name_to_code = {v: k for k, v in dx_map.items()}
    p = ensemble_df[dx_names].to_numpy()
    y = ensemble_df['actual'].map(name_to_code).to_numpy()
    y_hat = ensemble_df['predicted'].map(name_to_code).to_numpy()

    return p, y, y_hat, ensemble_df