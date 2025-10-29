"""
Exploratory visualization utility for class/metadata distributions and model prediction examples.

This module should be executed directly, and is intended for twofold dataset analysis:

    1. Pre-run: renders exploratory plots for class, age, lesion localization, and sex distributions.
    2. Post-run: renders representative lesion images using predicted ground-truth probabilities from a CSV file.
"""

import pandas as pd
import export, pipeline, plots

raw_df = pd.read_csv('../data/HAM10000_metadata')
train_df, val_df, test_df, _, dx_names = pipeline.load_data()
exp_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# Plot class distribution
def render_dx_dist():
    exp_counts = exp_df['dx'].value_counts()
    raw_counts = raw_df['dx'].value_counts()
    dropped = raw_counts['nv'] - exp_counts['nv']
    used = len(raw_df) - dropped

    fig = plots.plot_dx_dist(exp_counts, dropped, used, dx_names)
    export.save_fig(fig, 'dx_dist')

# Plot age distribution
def render_age_dist(bias_check=False):
    age_df = raw_df.copy()
    age_df = age_df.fillna(0.0) # Treat NaN as 0 (unknown age)

    # Optional sanity check for unknown age bias
    if bias_check:
        unknown = (age_df['age'] == 0.0).sum() # Count all unknown ages
        total = len(age_df)
        print(f'Age unknown for {unknown} of {total} samples...')
        for dx in dx_names:
            zeros = ((age_df['age'] == 0) & (age_df['dx'] == dx)).sum()
            subtotal = (age_df['dx'] == dx).sum()
            print(f'{dx}: {zeros} of {subtotal}')

    age_df = age_df[age_df['age'] > 0]

    fig = plots.plot_age_dist(age_df, dx_names)
    export.save_fig(fig, 'age_dist')

# Plot lesion localization and sex distributions
def render_meta_dist():
    plot_cfg = {
        'sex': {'palette': ['#440154', '#29AF7F', '#FDE725'], 'label': 'Sex'},
        'localization': {'palette': 'tab20', 'label': 'Lesion Localization'},
    }

    for category, cfg in plot_cfg.items():
        df_meta = raw_df.copy()
        palette = cfg['palette']
        title = cfg['label']

        df_meta = df_meta.groupby(['dx', category]).size().reset_index(name='count')
        df_meta['count'] = df_meta.groupby('dx')['count'].transform(lambda x: x / x.sum())

        fig = plots.plot_meta_dist(df_meta, category, palette, title, dx_names)
        export.save_fig(fig, f'{category}_dist')

# Plot representative lesion images
def render_images():
    df = pd.read_csv('../' + path)
    results = {'max': {}, 'med': {}, 'min': {}}
    img_dfs = {}

    for dx in dx_names:
        sub_df = df[df.actual == dx]
        med = sub_df[dx].median()
        med_idx = (sub_df[dx] - med).abs().idxmin()

        results['max'][dx] = sub_df.loc[sub_df[dx].idxmax(), 'image_id']
        results['med'][dx] = sub_df.loc[med_idx, 'image_id']
        results['min'][dx] = sub_df.loc[sub_df[dx].idxmin(), 'image_id']

    for name, id_map in results.items():
        img_dfs[name] = pd.DataFrame({'dx': id_map.keys(), 'image_id': id_map.values()})
        img_dfs[name] = pipeline.map_image_paths(img_dfs[name]).drop(columns=['image_id'])

    fig = plots.plot_images(img_dfs)
    export.save_fig(fig, 'lesions')

if __name__ == '__main__':
    mode = int(input('Enter rendering mode (1: pre-run, 2: post-run): '))

    if mode == 1:
        render_age_dist()
        render_dx_dist()
        render_meta_dist()
    elif mode == 2:
        path = input('Enter path to predictions CSV: ').strip()
        render_images()
    else:
        raise ValueError(f'Invalid rendering mode: {mode}')