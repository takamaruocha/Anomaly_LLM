import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

def generate_synthetic_timeseries():
    # 正常データ：正弦波 + 小ノイズ
    x = np.linspace(0, 30 * np.pi, 3000)
    y = np.sin(x) + 0.05 * np.random.randn(3000)

    # 異常部分（1500～1550）
    anomaly_start = 1500
    anomaly_end = 1550
    y[anomaly_start:anomaly_end] = 5.0 + 0.5 * np.random.randn(anomaly_end - anomaly_start)

    return y.reshape(-1, 1), [(anomaly_start, anomaly_end)]

def split_and_create_labels(series, anomalies, seq_len=512, forecast_horizon=96, initial_seq_start=0):
    """
    シリーズを分割し、対応する異常区間リストを作成
    """
    length = series.shape[0]
    label = np.zeros(length)
    for start, end in anomalies:
        label[start:end] = 1

    # 正規化 (全データ使う)
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
            break  # 長さが足りなければ打ち切り

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
        anomaly_list.append([anomaly_intervals])  # 単変量のセンサを1層包む

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
    output_dir = "/Storage2/maru/datasets/UCR_Anomaly_Archive/AnomLLM/train/Synthetic_SingleAnomaly_1024"
    seq_len = 1024
    forecast_horizon = 96
    initial_seq_start = 0  # 最初からスタート

    # データ生成
    series, anomalies = generate_synthetic_timeseries()

    # 分割と異常区間作成
    series_list, anomaly_list = split_and_create_labels(
        series,
        anomalies,
        seq_len=seq_len,
        forecast_horizon=forecast_horizon,
        initial_seq_start=initial_seq_start
    )

    # 保存
    save_as_data_pkl(output_dir, series_list, anomaly_list)

if __name__ == "__main__":
    main()

