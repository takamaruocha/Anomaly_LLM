import os
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler


def load_and_split_train_dataset(file_path, context_length=96, scale=True):
    """
    ファイルから訓練データ部分（正常）を読み込み、context_lengthごとに分割。
    異常はないため、全ての anomaly_intervals は [[]] に設定。
    """
    file_name = os.path.basename(file_path)
    fields = file_name.split('_')
    meta_data = {
        'name': '_'.join(fields[:4]),
        'train_end': int(fields[4]),
    }

    # ファイル読み込み
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
        Y_train = scaler.transform(Y_train)

    length = Y_train.shape[0]
    num_chunks = length // context_length

    all_segments = []
    all_anomalies = []

    for i in range(num_chunks):
        start_idx = i * context_length
        end_idx = start_idx + context_length
        if end_idx > length:
            break

        segment = Y_train[start_idx:end_idx].copy()
        all_segments.append(segment)
        all_anomalies.append([[]])  # 異常なし（正常区間）

    return all_segments, all_anomalies


def save_dataset_as_pickle(output_dir, series_list, anomaly_list):
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
    output_dir = "/Storage2/maru/datasets/UCR_Anomaly_Archive/AnomLLM/train/"
    context_length = 96

    files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]

    print(f"📂 処理対象ファイル数: {len(files)}")

    for i, file_name in enumerate(files):
        full_path = os.path.join(input_dir, file_name)

        try:
            # 訓練データ読み込み＆分割
            series_list, anomaly_list = load_and_split_train_dataset(
                full_path,
                context_length=context_length,
            )

            # 保存先ディレクトリ名（ファイル名の先頭4要素で）
            save_dir_name = '_'.join(file_name.split('_')[:4])
            save_dir = os.path.join(output_dir, save_dir_name)

            save_dataset_as_pickle(save_dir, series_list, anomaly_list)

        except Exception as e:
            print(f"❌ エラー（{file_name}）: {e}")


if __name__ == "__main__":
    main()

