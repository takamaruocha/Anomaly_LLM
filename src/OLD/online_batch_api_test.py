import os
import json
import argparse
from loguru import logger
from utils import generate_batch_AD_requests
from config import create_batch_api_configs
from openai_api import openai_client
from data.synthetic import SyntheticDataset


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process online batch API anomaly detection.')
    parser.add_argument('--variant', type=str, default='1shot-vision', help='Variant type')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='Model name')
    parser.add_argument('--data', type=str, default='point', help='Data name')
    return parser.parse_args()


def online_AD_batch(
    model_name: str,
    data_name: str,
    request_func: callable,
    variant: str = "standard",
    entity: str = "160_UCR_Anomaly_TkeepThirdMARS"
):
    base_dir = "/Storage2/maru/datasets/UCR_Anomaly_Archive/AnomLLM/"
    data_dir = os.path.join(base_dir, "eval", entity)
    train_dir = os.path.join(base_dir, "train", entity)
    results_dir = f'results/{data_name}/{entity}/{model_name}'
    os.makedirs(results_dir, exist_ok=True)

    jsonl_fn = os.path.join(results_dir, f"{variant}.jsonl")
    batch_fn = os.path.join(results_dir, f"{variant}_batch.json")

    eval_dataset = SyntheticDataset(data_dir)
    eval_dataset.load()

    train_dataset = SyntheticDataset(train_dir)
    train_dataset.load()

    # 1. JSONL ファイルを生成
    with open(jsonl_fn, 'w') as f:
        for i in range(len(eval_dataset)):
            custom_id = f"{data_name}_{model_name}_{variant}_{str(i).zfill(5)}"
            request = request_func(eval_dataset.series[i], train_dataset)
            f.write(json.dumps({
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": request
            }) + '\n')

    # 2. OpenAI クライアント生成
    client = openai_client(model_name)

    # 3. JSONL をアップロードしてバッチ作成
    batch_input_file = client.files.create(
        file=open(jsonl_fn, "rb"),
        purpose="batch"
    )
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"AD batch for {entity}"}
    )

    # 4. バッチ情報を保存
    with open(batch_fn, 'w') as f:
        json.dump(batch, f, default=lambda o: o.__dict__, indent=2)

    logger.info(f"✅ Created batch {batch.id} for {entity} / {variant}")


def main():
    args = parse_arguments()
    batch_api_configs = create_batch_api_configs()
    request_func = batch_api_configs[args.variant]

    # entity一覧を取得（evalファイル名から）
    datasets = ["005_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature1_4000_5391_5392.txt", "113_UCR_Anomaly_CIMIS44AirTemperature1_4000_5391_5392.txt"]
    
    for dataset in datasets:
        entity = '_'.join(dataset.split('_')[:4])
        print(f"\n📦 Generating batch for: {entity}")

        online_AD_batch(
            model_name=args.model,
            data_name=args.data,
            request_func=request_func,
            variant=args.variant,
            entity=entity,
        )


if __name__ == '__main__':
    main()

