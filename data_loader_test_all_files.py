import os
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler


def load_and_split_dataset(file_path, context_length=96, scale=True, seq_start=1120):
    """
    指定ファイルからテスト時系列データを読み込み、seq_start以降をcontext_lengthで分割。
    各分割ウィンドウに対応する異常区間を (start, end) タプルで記録。
    """
    file_name = os.path.basename(file_path)
    fields = file_name.split('_')
    meta_data = {
        'name': '_'.join(fields[:4]),
        'train_end': int(fields[4]),
        'anomaly_start_in_test': int(fields[5]) - int(fields[4]),
        'anomaly_end_in_test': int(fields[6][:-4]) - int(fields[4]),
    }

    with open(file_path) as f:
        Y = f.readlines()
        if len(Y) == 1:
            Y = Y[0].strip()
            Y = np.array([eval(y) for y in Y.split(" ") if len(y) > 1]).reshape((1, -1))
        elif len(Y) > 1:
            Y = np.array([eval(y.strip()) for y in Y]).reshape((1, -1))

    Y = Y.reshape(-1, 1)
    Y_train = Y[:meta_data['train_end']]

    if scale:
        scaler = StandardScaler()
        scaler.fit(Y_train)
        Y = scaler.transform(Y)

    Y_test = Y[meta_data['train_end']:]
    length = Y_test.shape[0]

    if seq_start >= length:
        raise ValueError(f"seq_start={seq_start} はテストデータの長さ {length} を超えています。")

    label = np.zeros(length)
    label[meta_data['anomaly_start_in_test']:meta_data['anomaly_end_in_test'] + 1] = 1

    num_chunks = (length - seq_start) // context_length
    all_segments = []
    all_anomalies = []

    for i in range(num_chunks):
        start_idx = seq_start + i * context_length
        end_idx = start_idx + context_length
        segment = Y_test[start_idx:end_idx].copy()
        segment_label = label[start_idx:end_idx]

        anomaly_intervals = []
        in_anomaly = False
        for j in range(context_length):
            if segment_label[j] == 1 and not in_anomaly:
                start = j
                in_anomaly = True
            elif segment_label[j] == 0 and in_anomaly:
                anomaly_intervals.append((start, j))
                in_anomaly = False
        if in_anomaly:
            anomaly_intervals.append((start, context_length))

        all_segments.append(segment)
        all_anomalies.append([anomaly_intervals])  # 単変量センサとして1センサぶん包む

    return all_segments, all_anomalies


def save_dataset_as_pickle(output_dir, series_list, anomaly_list):
    """
    seriesとanomを辞書にまとめてdata.pklとして保存
    """
    os.makedirs(output_dir, exist_ok=True)
    data_dict = {
        'series': series_list,
        'anom': anomaly_list
    }
    output_path = os.path.join(output_dir, 'data.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"✅ 保存: {output_path}")


def main():
    input_dir = "/Storage2/maru/datasets/UCR_Anomaly_Archive/UCR_Anomaly_FullData/"
    output_dir = "/Storage2/maru/datasets/UCR_Anomaly_Archive/AnomLLM/eval/"
    context_length = 96
    seq_start = 1120

    files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]

    print(f"📂 処理対象ファイル数: {len(files)}")

    for i, file_name in enumerate(files):
        full_path = os.path.join(input_dir, file_name)

        try:
            # データ読み込み＆分割
            series_list, anomaly_list = load_and_split_dataset(
                full_path,
                context_length=context_length,
                seq_start=seq_start
            )

            # 保存先ディレクトリ名（ファイル名の先頭4要素で）
            save_dir_name = '_'.join(file_name.split('_')[:4])
            save_dir = os.path.join(output_dir, save_dir_name)

            save_dataset_as_pickle(save_dir, series_list, anomaly_list)

        except Exception as e:
            print(f"❌ エラー（{file_name}）: {e}")


if __name__ == "__main__":
    main()

