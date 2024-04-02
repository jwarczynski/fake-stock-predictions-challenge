from glob import glob
import pandas as pd


def read_to_df(root):
    df = pd.DataFrame()

    for name in glob(f'{root}/**/*.csv', recursive=True):
        path_parts = name.split('\\')
        scalar_funcs = path_parts[-3].split('_')[-2]
        generations = path_parts[-3].split('_')[-1]
        run_id = path_parts[-2].split('_')[-1]
        generation = path_parts[-1].split('_')[-1].split('.')[0]

        data = pd.read_csv(name, skiprows=1, names=['Profit', 'Risk', 'Diversity'])
        data['Generation'] = generation
        data['Run_id'] = run_id
        data['Scalar_funcs'] = scalar_funcs
        data['TotalGenerations'] = generations

        df = pd.concat([df, data], ignore_index=True)

    return df
