import os
import json

def get_scores_mean_from_nested_folders(root_path):
    # スコアを格納する辞書
    score_lists = {
        "precision": [],
        "recall": [],
        "f1": [],
        "affi_precision": [],
        "affi_recall": [],
        "affi_f1": [],
    }

    # フォルダ内を探索
    for folder_name in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder_name)

        if os.path.isdir(folder_path):
            json_file_path = os.path.join(folder_path, 'output_results.json')
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r') as file:
                    data = json.load(file)

                    for key in score_lists.keys():
                        if key in data:
                            score_lists[key].append(data[key])

    # 平均を計算して表示
    for key, values in score_lists.items():
        if values:
            mean_val = sum(values) / len(values)
            print(f"{key} の平均: {mean_val:.5f}")
        else:
            print(f"{key} の値が見つかりませんでした。")

# 使用するルートディレクトリを指定
root_path = '/Storage2/maru/models/AnomLLM/0shot-text/'
get_scores_mean_from_nested_folders(root_path)

