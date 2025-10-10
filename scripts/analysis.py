import pandas as pd
import export, pipeline, plots

raw_df = pd.read_csv('../data/HAM10000_metadata')
train_df, val_df, test_df, _, dx_names = pipeline.load_data()
exp_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

def explore_age_dist():
    age_df = raw_df.copy()

    # Count both NaN and 0 values as 'unknown'
    age_df = age_df.fillna(0.0)
    unknown = (age_df['age'] == 0.0).sum()

    total = len(age_df)
    age_df = age_df[age_df['age'] > 0]

    print(f'Age unknown for {unknown} of {total} samples.')

    plot = plots.plot_age_dist(age_df, dx_names)
    export.save_eda(plot, 'age_dist')

def explore_dx_dist():
    exp_counts = exp_df['dx'].value_counts()
    raw_counts = raw_df['dx'].value_counts()
    dropped = raw_counts['nv'] - exp_counts['nv']
    used = len(raw_df) - dropped

    plot = plots.plot_dx_dist(exp_counts, dropped, used, dx_names)
    export.save_eda(plot, 'dx_dist')

def explore_meta_dist():
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

        plot = plots.plot_meta_dist(df_meta, category, palette, title, dx_names)
        export.save_eda(plot, f'{category}_dist')

def explore_images():
    df = pd.read_csv('../' + path)
    results = {'max_tp': {}, 'min_tp': {}, 'min_fn': {}}
    img_dfs = {}

    for dx in dx_names:
        tps = df[(df.actual == dx) & (df.predicted == dx)]
        fns = df[(df.actual == dx) & (df.predicted != dx)]

        results['max_tp'][dx] = tps.loc[tps[dx].idxmax(), 'image_id']
        results['min_tp'][dx] = tps.loc[tps[dx].idxmin(), 'image_id']
        results['min_fn'][dx] = fns.loc[fns[dx].idxmin(), 'image_id']

    for name, id_map in results.items():
        img_dfs[name] = pd.DataFrame({'dx': id_map.keys(), 'image_id': id_map.values()})
        img_dfs[name] = pipeline.map_image_paths(img_dfs[name]).drop(columns=['image_id'])

    plot = plots.plot_images(img_dfs)
    export.save_eda(plot, 'lesions')

if __name__ == '__main__':
    mode = int(input('Enter analysis mode (0 for EDA, 1 for post-run): '))

    if mode == 0:
        explore_age_dist()
        explore_dx_dist()
        explore_meta_dist()
    elif mode == 1:
        path = input('Enter path to predictions CSV: ').strip()
        explore_images()
    else:
        raise ValueError(f'Invalid analysis mode: {mode}')