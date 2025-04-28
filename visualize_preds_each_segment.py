import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

def plot_time_series(trues, test_labels, anomaly_results, start_time, end_time, y_min, y_max, save_path=None):
    if end_time is None:
        end_time = len(trues)  # デフォルトで最後まで表示

    fig, axs = plt.subplots(1, 1, figsize=(5, 2), sharex=True)
    x_range = np.arange(start_time, end_time)

    # ハイライトのインデックスを取得
    correct_detection = (test_labels == 1) & (anomaly_results == 1)  # 実際の異常を正しく検知
    missed_detection = (test_labels == 1) & (anomaly_results == 0)  # 実際の異常を見逃し
    false_alarm = (test_labels == 0) & (anomaly_results == 1)  # 誤検知

    for i in range(start_time, end_time):
        if correct_detection[i] == 1:  # 実際の異常を正しく検知
            axs.axvline(i, color='plum', alpha=0.2, zorder=0)
        if missed_detection[i] == 1:  # 実際の異常を見逃し 
            axs.axvline(i, color='lavenderblush', alpha=0.4, zorder=0)
        if false_alarm[i] == 1:  # 誤検知
            axs.axvline(i, color='steelblue', alpha=0.05, zorder=0)

    # 上段: オリジナルと復元のプロット
    axs.plot(x_range, trues[start_time:end_time], label='Ground Truth', color='dimgray', linewidth=0.8)

    # vus_roc = vus_roc * 100
    axs.set_title(f'Ground Truth', fontsize=12)
    axs.set_ylabel('Value', fontsize=11)
    axs.set_ylim([y_min-0.5, y_max+0.5])
    axs.legend(fontsize=11, loc='upper left')

    # 軸のラベルとタイトルの文字サイズの設定
    axs.tick_params(axis='both', which='major', labelsize=9)

    # 画像を保存
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"グラフを {save_path} に保存しました。")


def main(args=None):
    datasets = ["Synthetic_SingleAnomaly"]
    # datasets = ["Easy_Synthetic_SingleAnomaly"]
    data_name = "synthetic_timeseries"

    for dataset in datasets:
        fields = dataset.split('_')
        entity = '_'.join(fields[:4])

        variant = "0shot-text"
        variant_dir = f"performance_results/{data_name}/{variant}/{entity}"

        # ファイル読み込み
        test_labels = np.load(os.path.join(variant_dir, "gts.npy")).flatten()
        anomaly_results = np.load(os.path.join(variant_dir, "preds.npy")).flatten()

        # 元データ読み込み
        original_path = f"datasets/eval/{entity}/data.pkl"
        with open(original_path, "rb") as f:
            data = pickle.load(f)
        series_list = data["series"]
        print(len(series_list), series_list[0].shape)

        save_dir = f"visualization/{data_name}/{variant}/{entity}"
        os.makedirs(save_dir, exist_ok=True)

        all_series = np.vstack(series_list)
        global_min = np.min(all_series)
        global_max = np.max(all_series)

        # データ数
        num_series = len(series_list)

        # (データ数, 96) にreshape
        test_labels = test_labels.reshape(num_series, 96)
        anomaly_results = anomaly_results.reshape(num_series, 96)

        print(num_series, test_labels.shape, anomaly_results.shape)

        for idx in range(num_series):
            # 各時系列 (512, 1) → (512,) に変換
            series = series_list[idx].flatten()

            # ラベル (416 zeros + 後半96個) を作成
            padded_test_labels = np.concatenate([np.zeros(512 - 96), test_labels[idx]])
            padded_anomaly_results = np.concatenate([np.zeros(512 - 96), anomaly_results[idx]])

            save_path = os.path.join(save_dir, f"segment_visualization_{idx:03d}.png")

            plot_time_series(
                trues=series,
                test_labels=padded_test_labels,
                anomaly_results=padded_anomaly_results,
                start_time=0,
                end_time=512,
                y_min=global_min,
                y_max=global_max,
                save_path=save_path
            )


if __name__ == "__main__":
    main()
