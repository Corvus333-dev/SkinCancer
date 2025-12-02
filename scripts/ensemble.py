import itertools
import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, f1_score

from config import BASE_MODELS

# Create project root path relative to this module
ROOT = Path(__file__).resolve().parent.parent

def pool_models():
    """
    Collects paths to the most recent subexperiment for each model variant.

    Expected directory structure:

        /models/
        ├──backbone_1/ (e.g., ResNet50)
        │  ├──experiment_1/ (e.g., alpha_05_gamma_20)
        │  │  ├──20251130_2111
        │  │  └──YYYYMMDD_HHMM (i.e., latest timestamp)
        │  └──experiment_n/
        └──backbone_n/

    Returns:
        list: Path objects pointing to final experiment directories.

    """
    top_dir = Path(ROOT / 'models')
    model_pool = []

    for backbone_dir in top_dir.iterdir():
        if backbone_dir.name not in BASE_MODELS.keys():
            continue
        for variant_dir in backbone_dir.iterdir():
            exp_dirs = sorted(variant_dir.iterdir())
            if exp_dirs:
                model_pool.append(exp_dirs[-1])

    return model_pool

def load_predictions(model_pool, dx_names, source='val'):
    """
    Loads a predictions CSV for each experiment directory in the model pool and caches the results as DataFrames with
    model-specific column name suffixes.

    Args:
        model_pool (list or tuple): Path objects pointing to directories containing predictions CSV files.
        dx_names (list): Diagnosis names.
        source (str): Predictions source ('val' or 'test').

    Returns:
        dict: Map of experiment directory (Path) to predicted probability distributions (DataFrame).

    """
    cache = {}

    for exp_dir in model_pool:
        tag = exp_dir.name
        df = pd.read_csv(exp_dir / f'{source}_predictions.csv')
        df = df.drop(columns='predicted')
        df = df.rename(columns={dx_name: f'{dx_name}_{tag}' for dx_name in dx_names})
        cache[exp_dir] = df

    return cache

def merge_predictions(cache, combo):
    """
    Merges predicted probability distributions from multiple models on 'image_id', keeping the ground-truth column from
    the first DataFrame and dropping it from the rest.

    Args:
        cache (dict): Map of experiment directory (Path) to predicted probability distributions (DataFrame).
        combo (tuple): Path objects corresponding to optimal model subset.

    Returns:
        pd.DataFrame: Merged DataFrame containing per-model probability distributions and ground truths.
    """
    dfs = [cache[exp_dir] for exp_dir in combo]
    left_df = dfs[0]

    for df in dfs[1:]:
        left_df = left_df.merge(df.drop(columns='actual'), on='image_id')

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

def optimize_ensemble(cache, dx_map, dx_names, r=4):
    """
    Exhaustively evaluates all size-r model subsets and selects the combination that maximizes Cohen's kappa score.

    Computational complexity grows combinatorially:
        C(n, r) = n! / r! / (n - r)!

    Args:
        cache (dict): Map of experiment directory (Path) to predicted probability distributions (DataFrame).
        dx_map (dict): Map of diagnosis codes to diagnosis names.
        dx_names (list): Diagnosis names.
        r (int): Length of the model combination sequence.

    Returns:
        tuple: Path objects corresponding to optimal model subset.
    """
    best_ensemble = None
    best_score = float('-inf')

    combos = list(itertools.combinations(cache.keys(), r))
    t = len(combos)

    for i, combo in enumerate(combos, start=1):
        merged_df = merge_predictions(cache, combo)
        ensemble_df = ensemble_models(merged_df, dx_names)
        _, y, y_hat, _ = unpack_predictions(ensemble_df, dx_map, dx_names)

        score = cohen_kappa_score(y, y_hat)

        if score > best_score:
            best_score = score
            best_ensemble = combo

        progress = (i / t) * 100
        print(f'\r[{i}/{t}] | {progress:4.1f}%', end='', flush=True)

    return best_ensemble

# Stage ensemble creation
def compute_ensemble(dx_map, dx_names, mode='test'):
    model_pool = pool_models()
    cache = load_predictions(model_pool, dx_names, source='val')
    best_ensemble = optimize_ensemble(cache, dx_map, dx_names)

    cache = load_predictions(best_ensemble, dx_names, source=mode)
    merged_df = merge_predictions(cache, best_ensemble)
    ensemble_df = ensemble_models(merged_df, dx_names)
    p, y, y_hat, ensemble_df = unpack_predictions(ensemble_df, dx_map, dx_names)

    return p, y, y_hat, ensemble_df, best_ensemble