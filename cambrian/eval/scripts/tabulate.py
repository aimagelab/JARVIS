import os
import json
import pandas as pd
import argparse
from math import floor
from pathlib import Path


def tabulate_results(eval_dir, experiment_csv_fname, out_pivot_fname, out_all_results_fname):
    exists = os.path.exists(eval_dir)
    if not exists:
        raise ValueError(f"eval_dir {eval_dir} does not exist")

    print(f"eval_dir: {eval_dir}")

    evals_order = [
        # llava
        # 'vqav2',
        'gqa',
        'vizwiz',
        'scienceqa',
        'textvqa',
        'pope',
        'mme',
        'mmbench_en',
        'mmbench_cn',
        'seed',
        # 'llava_w',
        # 'mmvet', # submission
        # Addtl
        'mmmu',
        'mathvista',
        'ai2d',
        'chartqa',
        # 'docvqa', # submission
        # 'infovqa', # submission
        # 'stvqa', # submission
        'ocrbench',
        'mmstar',
        'realworldqa',
        'qbench',
        'blink',
        'mmvp',
        'vstar',
        'ade',
        'omni',
        'coco'
        # 'synthdog', # seems broken?
    ]

    evals_col_overrides = {
        'scienceqa': '100x_multimodal_acc',
        'mme': "Perception",
        'mmbench_en': "100x_circular_accuracy",
        'mmbench_cn': "100x_circular_accuracy",
        'seed': "100x_accuracy",
        'mmmu': "100x_accuracy",
        'mathvista': "100x_accuracy",
        'ocrbench': "total_accuracy['accuracy']",
        'qbench': "100x_accuracy",
        'blink': "100x_accuracy",
        'ade': "100x_accuracy",
        'omni': "100x_accuracy",
        'coco': "100x_accuracy",
    }

    cols_breakdown = {
        'ade': ['Count', 'Relative location'],
        'coco': ['Count', 'Relative location'],
        'omni': ['Depth', 'Distance'],
        'blink': ['Counting', 'IQ_Test', 'Object_Localization', 'Relative_Depth', 'Relative_Reflectance', 'Spatial_Relation']
    }

    df_blink = None
    df_coco = None
    df_ade = None
    df_cvbench3d = None

    # gather results from each eval
    dfs = []
    for eval_name in evals_order:
        results_path = os.path.join(eval_dir, eval_name, experiment_csv_fname)
        if not os.path.exists(results_path):
            print(f"Skipping {eval_name} as no results file found")
            continue

        try:
            df = pd.read_csv(results_path)
        except Exception as e:
            print(f"Error reading {results_path}: {e}")
            raise e

        if eval_name in evals_col_overrides:
            override = evals_col_overrides[eval_name]
            if override.startswith("100x_"):
                override = override[5:]
                df["accuracy"] = df[override] * 100
            elif override == "total_accuracy['accuracy']":
                df["accuracy"] = df["total_accuracy"].apply(
                    lambda x: json.loads(x.replace("'", '"'))["accuracy"])
            else:
                df["accuracy"] = df[override]

        df["eval_name"] = eval_name

        df = df.sort_values("time")
        df = df.drop_duplicates("model", keep="last")

        if eval_name in cols_breakdown:
            if eval_name == 'blink':
                df_blink = df.copy()
                for col in cols_breakdown[eval_name]:
                    df_blink[col] = df[col].apply(
                        lambda x: eval(x)['accuracy']) * 100
            elif eval_name == 'ade':
                df_ade = df.copy()
                df_ade['Count'] = df['Count'].apply(lambda x: eval(x))
                df_ade['Relative location'] = df['Relative location'].apply(
                    lambda x: eval(x))
            elif eval_name == 'coco':
                df_coco = df.copy()
                df_coco['Count'] = df['Count'].apply(lambda x: eval(x))
                df_coco['Relative location'] = df['Relative location'].apply(
                    lambda x: eval(x))
            elif eval_name == 'omni':
                df_cvbench3d = df.copy()
                for col in cols_breakdown[eval_name]:
                    df_cvbench3d[col] = df[col].apply(
                        lambda x: eval(x)['accuracy']) * 100

        df = df[["time", "eval_name", "model", "accuracy"]]
        dfs.append(df)

    all_results = pd.concat(dfs)
    all_results.sort_values("time").to_csv(out_all_results_fname, index=False)
    print(f"Saved all results to {out_all_results_fname}")

    pivot = all_results.pivot(
        index="model", columns="eval_name", values="accuracy")
    completed_cols = [col for col in evals_order if col in pivot.columns]
    pivot = pivot[completed_cols]

    # if .xlsx, to_excel, else to_csv
    if out_pivot_fname.endswith(".xlsx"):
        pivot.to_excel(out_pivot_fname)
        print(f"Saved excel file pivot to {out_pivot_fname}")
    else:
        pivot.to_csv(out_pivot_fname)
        print(f"Saved csv file pivot to {out_pivot_fname}")

    out_pivot_fname = Path(out_pivot_fname)
    if df_blink is not None:
        fname = out_pivot_fname.parent / f"blink_{out_pivot_fname.stem}.csv"
        df_blink[['model', 'accuracy', *cols_breakdown['blink']]
                 ].to_csv(fname, index=False)
        print(f"Saved Blink results breakdown to {fname}")

    if df_ade is not None and df_coco is not None and df_cvbench3d is not None:
        fname = out_pivot_fname.parent / f"cvbench_{out_pivot_fname.stem}.csv"
        df_cvbench = df_ade.merge(df_coco, on='model').merge(
            df_cvbench3d, on='model')
        df_cvbench['Count 2D'] = df_cvbench.apply(lambda row: (floor(row['Count_x']['accuracy'] * row['Count_x']['total']) + floor(
            row['Count_y']['accuracy'] * row['Count_y']['total'])) / (row['Count_x']['total'] + row['Count_y']['total']) * 100, axis=1)
        df_cvbench['Relative location 2D'] = df_cvbench.apply(lambda row: (floor(row['Relative location_x']['accuracy'] * row['Relative location_x']['total']) + floor(
            row['Relative location_y']['accuracy'] * row['Relative location_y']['total'])) / (row['Relative location_x']['total'] + row['Relative location_y']['total']) * 100, axis=1)
        df_cvbench.rename(columns={'Depth': 'Depth 3D', 'Distance': 'Distance 3D', 'accuracy': 'Accuracy 3D',
                          'accuracy_x': 'Accuracy ADE', 'accuracy_y': 'Accuracy COCO'}, inplace=True)
        df_cvbench = df_cvbench[['model', 'Count 2D', 'Relative location 2D',
                                 'Depth 3D', 'Distance 3D', 'Accuracy 3D', 'Accuracy ADE', 'Accuracy COCO']]
        df_cvbench['Accuracy 2D'] = (
            df_cvbench['Accuracy ADE'] + df_cvbench['Accuracy COCO']) / 2
        df_cvbench.to_csv(fname, index=False)
        print(f"Saved CV-Bench results breakdown to {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tabulate experiment results")
    parser.add_argument("--eval_dir", type=str, default="eval",
                        help="Directory containing evaluation results")
    parser.add_argument("--experiment_csv", type=str, default="experiments.csv",
                        help="Name of the CSV file containing experiment results")
    parser.add_argument("--out_pivot", type=str, default="pivot.xlsx",
                        help="Name of the output file (Excel or CSV)")
    parser.add_argument("--out_all_results", type=str, default="all_results.csv",
                        help="Name of the CSV file to save all results")

    args = parser.parse_args()

    tabulate_results(args.eval_dir, args.experiment_csv,
                     args.out_pivot, args.out_all_results)
