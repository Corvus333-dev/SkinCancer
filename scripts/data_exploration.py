import pandas as pd

import export, plots, pipeline

train_df, val_df, test_df, dx_map, dx_names = pipeline.load_data()
exp_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

def explore_age_dist():
    df_age = exp_df.copy()
    df_age['age'] *= 100 # Reverse normalization
    df_age = df_age[df_age['age'] > 0] # Remove sentinel values

    return plots.plot_age_dist(df_age, dx_names)

def explore_dx_dist():
    raw_df = pd.read_csv('../data/HAM10000_metadata')

    exp_counts = exp_df['dx'].value_counts()
    raw_counts = raw_df['dx'].value_counts()
    dropped = raw_counts['nv'] - exp_counts['nv']
    used = len(raw_df) - dropped

    return plots.plot_dx_dist(exp_counts, dropped, used, dx_names)

if __name__ == '__main__':
    age_dist_plot = explore_age_dist()
    dx_dist_plot = explore_dx_dist()
    export.save_eda(age_dist_plot, dx_dist_plot)