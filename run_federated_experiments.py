from dotenv import load_dotenv
load_dotenv()

from utils.utils import append_dict_to_json
from config_folder import client_config_file, model_config
from src.federated_learning.server_setup import run_server


if __name__=="__main__":
    # model_name = "meta-llama/Llama-3.1-8B-Instruct"
    num_cpus = 4
    num_gpus = 3
    # models = ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]
    # strategies = ['FedAvg', 'FedProx', 'Drift', 'FedAdam', 'FedTrimmedAvg']
    
    server_rounds = 5
    lora_rank=16
    for random_state in [42]: #client_config_file.TEST_RANDOM_STATES
        for model_name in [client_config_file.MODEL_NAMES[0]]:
            if model_name=="meta-llama/Llama-3.1-8B-Instruct":
                thought_data_path = "data/thought_data/llama/"
            if model_name=="Qwen/Qwen2.5-7B-Instruct":
                thought_data_path = "data/thought_data/qwen/"
            for strategy in ['Drift']: #client_config_file.STRATEGIES
                run_server(
                    num_cpus,
                    num_gpus,
                    gpu_id=None,
                    batch_size=4,
                    server_rounds=server_rounds,
                    quantization="4bit",
                    strategy_name=strategy,
                    thought_data_path=thought_data_path,
                    model_name = model_name,
                    lora_rank=lora_rank,
                    working_directory="/drift/",
                    random_state=random_state
                )
                training_log = {
                    "model_name": model_name,
                    "server_strategy": strategy,
                    "server_rounds": server_rounds,
                    "lora_rank": lora_rank,
                    "random_state": random_state,
                }
                append_dict_to_json(
                    file_path="drift/server_logs.json",
                    new_data=training_log
                )
