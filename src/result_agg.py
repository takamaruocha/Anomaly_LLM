import argparse
import numpy as np
import pandas as pd
from tqdm import trange
from prompt import time_series_to_image
from utils import (
    view_base64_image,
    display_messages,
    collect_results,
    plot_series_and_predictions,
    interval_to_vector,
    compute_metrics,
    process_dataframe,
    highlight_by_ranking,
    styled_df_to_latex,
)
import pickle
import os
from data.synthetic import SyntheticDataset


def load_datasets(entity):
    base_dir = "/Storage2/maru/datasets/UCR_Anomaly_Archive/AnomLLM/"
    data_dir = os.path.join(base_dir, "eval", entity)
    train_dir = os.path.join(base_dir, "train", entity)
    eval_dataset = SyntheticDataset(data_dir)
    eval_dataset.load()
    train_dataset = SyntheticDataset(train_dir)
    train_dataset.load()
    return eval_dataset, train_dataset


def compute_metrics_for_results(eval_dataset, results, num_samples=69):
    metric_names = [
        "precision",
        "recall",
        "f1",
        "affi precision",
        "affi recall",
        "affi f1",
    ]
    results_dict = {key: [[] for _ in metric_names] for key in results.keys()}

    for i in trange(0, num_samples):
        anomaly_locations = eval_dataset[i][0].numpy()
        gt = interval_to_vector(anomaly_locations[0])
        
        for name, prediction in results.items():
            try:
                metrics = compute_metrics(gt, prediction[i])
            except IndexError:
                print(f"experiment {name} not finished")
            for idx, metric_name in enumerate(metric_names):
                results_dict[name][idx].append(metrics[metric_name])

    df = pd.DataFrame(
        {k: np.mean(v, axis=1) for k, v in results_dict.items()},
        index=["precision", "recall", "f1", "affi precision", "affi recall", "affi f1"],
    )
    return df
"""


def compute_metrics_for_results(eval_dataset, results, num_samples):
    metric_names = [
        "precision",
        "recall",
        "f1",
        "affi precision",
        "affi recall",
        "affi f1",
    ]
    results_dict = {key: [[] for _ in metric_names] for key in results.keys()}

    for name, prediction in results.items():
        print(f"🧪 Evaluating {name})")
        gts = []
        preds = []

        for i in trange(0, num_samples):
            anomaly_locations = eval_dataset[i][0].numpy()
            gt = interval_to_vector(anomaly_locations[0])
            gts.append(gt)

            if prediction[i] is None:
                #print(f"[SKIP] {name} index {i} is missing or None")
                preds.append(np.zeros(len(gt)))
            else:
                preds.append(prediction[i].flatten())
                #print('pred: ', prediction[i])

        gts = np.concatenate(gts, axis=0)
        preds = np.concatenate(preds, axis=0)
        metrics = compute_metrics(gts, preds)
        for idx, metric_name in enumerate(metric_names):
            results_dict[name][idx].append(metrics[metric_name])

    df = pd.DataFrame(
        {k: np.mean(v, axis=1) for k, v in results_dict.items()},
        index=["precision", "recall", "f1", "affi precision", "affi recall", "affi f1"],
    )
    return df
"""

def main(args):
    # 使いたいデータセット一覧
    #entities = [
    #    "005_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature1",
    #    "113_UCR_Anomaly_CIMIS44AirTemperature1"
    #]
    root_path = "/Storage2/maru/datasets/UCR_Anomaly_Archive/UCR_Anomaly_FullData/"
    datasets = os.listdir(root_path)
    datasets = sorted(datasets, key=lambda x: int(x.split('_')[0]))

    data_name = "UCR_Anomaly"

    for dataset in datasets:
        fields = dataset.split('_')
        entity = '_'.join(fields[:4])

        label_name = f"label-{entity}"
        table_caption = f"Evaluation on {entity}"

        print(f"\n📊 Processing: {entity}")

        eval_dataset, train_dataset = load_datasets(entity)
        directory = f"results/{data_name}/{entity}"
        results = collect_results(directory, ignore=['phi'])

        df = compute_metrics_for_results(eval_dataset, results, len(eval_dataset))
        double_df = process_dataframe(df.T.copy())
        print(double_df)

        os.makedirs("results/agg", exist_ok=True)
        with open(f"results/agg/{entity}.pkl", "wb") as f:
            pickle.dump(double_df, f)

        styled_df = highlight_by_ranking(double_df)
        latex_table = styled_df_to_latex(styled_df, table_caption, label=label_name)
        print(latex_table)

        # LaTeXファイルに追記
        with open("out.tex", "a") as f:
            f.write(latex_table + "\n\n")

        # JSON 形式で variant ごとに保存
        for (model, variant), row in double_df.iterrows():
            # 保存先ディレクトリ
            variant_dir = os.path.join("/Storage2/maru/models/AnomLLM_per_segment/", variant, entity)
            os.makedirs(variant_dir, exist_ok=True)

            # 保存ファイル名（entityごと）
            output_path = os.path.join(variant_dir, f"output_results.json")

            # 指標を辞書化して保存
            metric_dict = {
                "model": str(model),
                "variant": str(variant),
                "entity": str(entity),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
                "affi_precision": float(row["affi precision"]),
                "affi_recall": float(row["affi recall"]),
                "affi_f1": float(row["affi f1"]),
             }

            with open(output_path, "w") as jf:
                import json
                json.dump(metric_dict, jf, indent=2)

            print(f"✅ Saved JSON: {output_path}")

"""
python src/result_agg.py --data_name trend --label_name trend-exp --table_caption "Trend anomalies in shifting sine wave"
python src/result_agg.py --data_name freq --label_name freq-exp --table_caption "Frequency anomalies in regular sine wave"
python src/result_agg.py --data_name point --label_name point-exp --table_caption "Point noises anomalies in regular sine wave"
python src/result_agg.py --data_name range --label_name range-exp --table_caption "Out-of-range anomalies in Gaussian noise"

python src/result_agg.py --data_name noisy-trend --label_name noisy-trend-exp --table_caption "Trend anomalies in shifting sine wave with extra noise"
python src/result_agg.py --data_name noisy-freq --label_name noisy-freq-exp --table_caption "Frequency anomalies in regular sine wave with extra noise"
python src/result_agg.py --data_name noisy-point --label_name noisy-point-exp --table_caption "Point noises anomalies in regular sine wave with Gaussian noise"
python src/result_agg.py --data_name flat-trend --label_name flat-trend-exp --table_caption "Trend anomalies, but no negating trend, and less noticeable speed changes"
"""  # noqa

if __name__ == "__main__":
    main(None)

