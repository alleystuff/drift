import os
import gc
import torch
from src.methods.baseline import baseline_train, baseline_evaluate

if __name__=="__main__":
    # manually run the following function to train and save mdoels for baselin
    # baseline_train(data_path="data/thought_data/qwen/")
    # gc.collect()
    # torch.cuda.empty_cache()
    # torch.cuda.ipc_collect()
    
    # manually run the following code to do evaluation on all tasks
    # set to only one device for evaluation to avoid deadlocks
    # os.environ['CUDA_VISIBLE_DEVICES']="7"
    # training_outputs_llama = [("meta-llama/Llama-3.1-8B-Instruct", "src/saved_models/baseline/llama_3_1_8B_instruct/baseline_40", 8, 40),
    #                     ("meta-llama/Llama-3.1-8B-Instruct", "src/saved_models/baseline/llama_3_1_8B_instruct/baseline_42", 8, 42),
    #                     ("meta-llama/Llama-3.1-8B-Instruct", "src/saved_models/baseline/llama_3_1_8B_instruct/baseline_48", 8, 48)]
    
    # [("Qwen/Qwen2.5-7B-Instruct", "src/saved_models/baseline/qwen_2_5_7B_instruct/baseline_40", 8, 40),
    # ("Qwen/Qwen2.5-7B-Instruct", "src/saved_models/baseline/qwen_2_5_7B_instruct/baseline_42", 8, 42)]
    training_outputs_qwen = [("Qwen/Qwen2.5-7B-Instruct", "src/saved_models/baseline/qwen_2_5_7B_instruct/baseline_48", 8, 48)]
    baseline_evaluate(training_outputs=training_outputs_qwen, results_filename="evaluation.json", evaluation_type="baseline", data_path="data/thought_data/qwen/")