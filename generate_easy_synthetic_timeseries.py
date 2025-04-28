import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

def generate_easy_synthetic_timeseries():
    """
    正常時はほぼフラット（定数）、異常は大きなスパイク
    """
    y = np.ones(3000) * 0.5  # ずっと0.5付近
    anomaly_start = 1500
    anomaly_end = 1550
    y[anomaly_start:anomaly_end] = 10.0  # 突然10にジャンプ

    return y.reshape(-1, 1), [(anomaly_start, anomaly_end)]

def split_and_create_labels(series, anomalies, seq_len=512, forecast_horizon=96, initial_seq_start=0):
    length = series.shape[0]
    label = np.zeros(length)
    for start, end in anomalies:
        label[start:end] = 1

    scaler = StandardScaler()
    scaler.fit(series)
    series = scaler.transform(series)

    num_chunks = (length - initial_seq_start) // forecast_horizon + 1
    series_list = []
    anomaly_list = []

    for i in range(num_chunks):
        seq_start = initial_seq_start + forecast_horizon * i
        seq_end = seq_start + seq_len

        if seq_end > length:
            break

        segment = series[seq_start:seq_end].copy()
        segment_label = label[seq_end-forecast_horizon:seq_end]

        anomaly_intervals = []
        in_anomaly = False
        for j in range(forecast_horizon):
            if segment_label[j] == 1 and not in_anomaly:
                start_idx = j
                in_anomaly = True
            elif segment_label[j] == 0 and in_anomaly:
                anomaly_intervals.append((start_idx, j))
                in_anomaly = False
        if in_anomaly:
            anomaly_intervals.append((start_idx, forecast_horizon))

        series_list.append(segment)
        anomaly_list.append([anomaly_intervals])

    return series_list, anomaly_list

def save_as_data_pkl(output_dir, series_list, anomaly_list):
    os.makedirs(output_dir, exist_ok=True)

    data = {
        "series": series_list,
        "anom": anomaly_list
    }

    output_path = os.path.join(output_dir, "data.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    print(f"✅ 保存完了: {output_path}")

def main():
    output_dir = "/Storage2/maru/datasets/UCR_Anomaly_Archive/AnomLLM/train/Easy_Synthetic_SingleAnomaly_1024"
    seq_len = 1024
    forecast_horizon = 96
    initial_seq_start = 0

    series, anomalies = generate_easy_synthetic_timeseries()

    series_list, anomaly_list = split_and_create_labels(
        series,
        anomalies,
        seq_len=seq_len,
        forecast_horizon=forecast_horizon,
        initial_seq_start=initial_seq_start
    )

    save_as_data_pkl(output_dir, series_list, anomaly_list)

if __name__ == "__main__":
    main()

