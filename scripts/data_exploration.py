import pandas as pd

import export, plots, pipeline

train_df, val_df, test_df, dx_map, dx_names = pipeline.load_data()
df = pd.concat([train_df, val_df, test_df], ignore_index=True)

def explore_age_dist():
    df_age = df.copy()
    df_age['age'] *= 100 # Reverse normalization
    df_age = df_age[df_age['age'] > 0] # Remove sentinel values
    return plots.plot_age_dist(df_age, dx_names)

def explore_dx_dist():
    # Undersampled dataset distribution
    return plots.plot_dx_dist(df, dx_names)

if __name__ == '__main__':
    age_dist_plot = explore_age_dist()
    dx_dist_plot = explore_dx_dist()
    export.save_eda(age_dist_plot, dx_dist_plot)