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
    base_dir = "datasets"
    data_dir = os.path.join(base_dir, "eval", entity)
    train_dir = os.path.join(base_dir, "train", entity)
    eval_dataset = SyntheticDataset(data_dir)
    eval_dataset.load()
    train_dataset = SyntheticDataset(train_dir)
    train_dataset.load()
    return eval_dataset, train_dataset


def compute_metrics_for_results(eval_dataset, results, num_samples):
    metric_names = [
        "precision",
        "recall",
        "f1",
        # "affi precision",
        # "affi recall",
        # "affi f1",
    ]
    results_dict = {key: [[] for _ in metric_names] for key in results.keys()}
    all_gts_preds = {}

    for name, prediction in results.items():
        # if name != "gpt-4o-2024-11-20 (0shot-text)":
        #     continue
        print(f"🧪 Evaluating {name}")
        gts = []
        preds = []

        print(len(eval_dataset), len(prediction))

        for i in trange(0, num_samples):
            # print(f"============{i}============")
            anomaly_locations = eval_dataset[i][0].numpy()
            gt = interval_to_vector(anomaly_locations[0], start=0, end=96)
            gts.append(gt)
            # print("gt: ", np.sum(gt))
            # print("gt: ", gt.shape)
            if prediction[i] is None:
                preds.append(np.zeros(len(gt)))
                # print("pred: ", "None")
                # print("pred: ", len(gt))
            else:
                preds.append(prediction[i].flatten())
                # print("pred", np.sum(prediction[i].flatten()))
                # print("pred: ", prediction[i].shape)

        gts = np.concatenate(gts, axis=0)
        preds = np.concatenate(preds, axis=0)
        print(gts.shape, preds.shape)
        all_gts_preds[name] = (gts, preds)

        metrics = compute_metrics(gts, preds)
        for idx, metric_name in enumerate(metric_names):
            results_dict[name][idx].append(metrics[metric_name])

    df = pd.DataFrame(
        {k: np.mean(v, axis=1) for k, v in results_dict.items()},
        index=["precision", "recall", "f1"]# , "affi precision", "affi recall", "affi f1"],
    )
    return df, all_gts_preds


def main(args):
    data_name = "synthetic_timeseries"

    # entity = "Synthetic_SingleAnomaly"
    entity = "Easy_Synthetic_SingleAnomaly"

    label_name = f"label-{entity}"
    table_caption = f"Evaluation on {entity}"

    print(f"\n📊 Processing: {entity}")

    eval_dataset, train_dataset = load_datasets(entity)
    # print("eval_dataset", len(eval_dataset), eval_dataset[0][0], eval_dataset[0][1].shape)
    directory = f"results/{data_name}/{entity}"
    results = collect_results(directory, ignore=['phi'])

    df, all_gts_preds = compute_metrics_for_results(eval_dataset, results, len(eval_dataset))
    double_df = process_dataframe(df.T.copy())
    print(double_df)

    # JSON 形式で variant ごとに保存
    for (model, variant), row in double_df.iterrows():
        # 保存先ディレクトリ
        variant_dir = os.path.join("performance_results", data_name, variant, entity)
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
            # "affi_precision": float(row["affi precision"]),
            # "affi_recall": float(row["affi recall"]),
            # "affi_f1": float(row["affi f1"]),
        }

        with open(output_path, "w") as jf:
            import json
            json.dump(metric_dict, jf, indent=2)

        print(f"✅ Saved JSON: {output_path}")

        key = f"{model} ({variant})"
        # if key == "gpt-4o-2024-11-20 (0shot-text)":
        gts, preds = all_gts_preds[key]
        np.save(os.path.join(variant_dir, "gts.npy"), gts)
        np.save(os.path.join(variant_dir, "preds.npy"), preds)

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

