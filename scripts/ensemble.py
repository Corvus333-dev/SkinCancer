import pandas as pd
from pathlib import Path

def merge_pred_dfs(best_models, dx_names):
    dfs = []

    for exp in best_models:
        exp_dir = Path(exp).parent
        timestamp = exp_dir.name
        df = pd.read_csv(exp_dir / 'val_predictions.csv')
        df = df.drop(columns='predicted')
        df = df.rename(columns={dx_name: f'{dx_name}_{timestamp}' for dx_name in dx_names})
        dfs.append(df)

    left_df = dfs[0]

    for df in dfs[1:]:
        df = df.drop(columns='actual')
        left_df = left_df.merge(df, on='image_id')

    return left_df

def ensemble_models(merged_df, dx_names):
    prob_cols = [c for c in merged_df.columns if any(c.startswith(dx_name) for dx_name in dx_names)]
    mean_probs = merged_df[prob_cols].groupby(lambda c: c.split('_')[0], axis=1).mean()
    mean_probs['predicted'] = mean_probs.idxmax(axis=1)
    ensemble_df = pd.concat([merged_df[['image_id', 'actual']], mean_probs], axis=1)
    cols = ['image_id'] + dx_names + ['actual', 'predicted']
    ensemble_df = ensemble_df[cols]

    return ensemble_df

def get_pred_results(ensemble_df, dx_map, dx_names):
    name_to_code = {v: k for k, v in dx_map.items()}
    p = ensemble_df[dx_names].to_numpy()
    y = ensemble_df['actual'].map(name_to_code).to_numpy()
    y_hat = ensemble_df['predicted'].map(name_to_code).to_numpy()

    return p, y, y_hat, ensemble_df