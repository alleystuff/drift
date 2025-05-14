from dotenv import load_dotenv
load_dotenv()

import gc
import torch
gc.collect()
torch.cuda.empty_cache()

import torch
import pandas as pd
pd.options.mode.chained_assignment = None

# from trl import DPOConfig, DPOTrainer
# from peft import get_peft_model, LoraConfig, PeftModel, PeftConfig
from peft import PeftModel, LoraConfig
from transformers import AutoModelForCausalLM, AutoProcessor #AutoTokenizer, TrainingArguments, 

from accelerate import Accelerator
from src.methods.dpo import get_dpo_trainer
from src.methods.evaluate import evaluate
from config_folder.model_config import MODELS
from config_folder.client_config_file import DATASETS, TEST_RANDOM_STATES
from utils.data_utils import get_train_test_split, get_preference_dataset
from utils.training_utils import get_quantization_config#, print_trainable_parameters, get_peft_parameters_and_peft_state_dict

def _baseline_train(
    model=None,
    model_name="Qwen/Qwen2.5-7B-Instruct",
    trained_model_save_path="src/saved_models/baseline/",
    data_path="data/thought_data/qwen/",
    device=None,
    lora_rank=8,
    lora_alpha=16,
    lora_dropout=0.05,
    random_state=40
):
    """This function will train the baseline model for CPO using all ToT data for all tasks.
    It will then save the model in the src/saved_models/baseline directory.

    Args:
        model (_type_, optional): Auto Model from huggingface. Defaults to None.
        model_name (str, optional): model name if model is not passed. Defaults to "meta-llama/Llama-3.1-8B-Instruct".
        trained_model_save_path (str, optional): this is where the baseline model will be saved. Defaults to "src/saved_models/baseline/".
        device (str, optional): gpu or cpu used for training. Defaults to "auto".
    """
    # params_dict = defaultdict()
    # tasks = list(DATASETS.keys())
    # models = list(MODELS.keys())
    if type(device)==int:
        # torch.cuda.set_device(device)
        # device_map = f"cuda:{device}"

        device_map = {"": Accelerator().local_process_index}
        print(device_map)
        # print({'':torch.cuda.current_device()})
    else:
        device_map = "auto"
    print(f"Device Map: {device_map}")
        
    if model==None:
        tokenizer = AutoProcessor.from_pretrained(
            model_name,
        )
        quantization_config = get_quantization_config(quantization="4bit")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            # load_in_4bit=True,
            use_cache=False,
        )

    # load train test splits and combine them all together
    tasks = list(DATASETS.keys())
    all_tasks_train_dataset, all_tasks_test_dataset = pd.DataFrame(), pd.DataFrame()
    for task in tasks:
        train_dataset, test_dataset = get_train_test_split(
            task_name=task,
            data_path=data_path,
            test_sample_size=0.2,
            test_random_state=random_state
        )
        all_tasks_train_dataset, all_tasks_test_dataset = pd.concat([all_tasks_train_dataset, train_dataset]), pd.concat([all_tasks_test_dataset, test_dataset])

    #set up preference dataset for training and testing using huggingface Dataset
    train_preference_dataset, test_preference_dataset = get_preference_dataset(all_tasks_train_dataset), get_preference_dataset(all_tasks_test_dataset)

    # define peft config
    # peft_config = LoraConfig(
    #     r=lora_rank,
    #     lora_alpha=lora_alpha,
    #     lora_dropout=lora_dropout,
    #     target_modules=[
    #         "q_proj",
    #         "v_proj",
    #         "k_proj",
    #     ],
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )
    
    friendly_model_name = MODELS[model_name]['model_name']
    output_dir = f"{trained_model_save_path}{friendly_model_name}/baseline_{random_state}"
    print(f"Model will be saved at: {output_dir}")
    trainer = get_dpo_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_preference_dataset,
        test_dataset=test_preference_dataset,
        batch_size=4,
        output_dir=output_dir,
        use_cpu=False,
        device=device
    )

    # train model and save it
    # trainer.model.to()
    # torch.cuda.
    # trainer._eval_dataloaders.to(device)
    # trainer.tra#.to(device)
    # trainer.args.device = "cuda:7"
    print(f"Training the model on device: {trainer.args.device}")
    trainer.train()
    trainer.save_model(output_dir)
    print(f"Model training done. Model saved at: {output_dir}")

    #clean up
    model = model.cpu()
    trainer_model = trainer.model.cpu()
    del tokenizer, model, trainer, trainer_model, train_preference_dataset, test_preference_dataset, train_dataset, test_dataset, quantization_config
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    # rerturn directory of the trained model
    return output_dir, lora_rank

def baseline_train(data_path="data/thought_data/qwen/"):
    """
    Train and evaluate models for baseline.
    """    
    import time
    model_names = list(MODELS.keys())
    
    # index on 0 just to train and evaluate LLama
    model_names = [model_names[1]] 
    
    for model_name in model_names:
        print(f"Training and Evaluating model: {model_name}")
        training_outputs = list()
        for random_state in TEST_RANDOM_STATES:
            print("Entering Model Training")
            model_and_tokenizer_save_path, lora_rank = _baseline_train(
                model_name=model_name,
                trained_model_save_path="src/saved_models/baseline/",
                data_path=data_path,
                device=None,
                lora_rank=8,
                lora_alpha=16,
                lora_dropout=0.05,
                random_state=random_state
            )
            training_outputs.append((model_name, model_and_tokenizer_save_path, lora_rank, random_state))
            # model_and_tokenizer_save_path = "src/saved_models/baseline/llama_3_1_8B_instruct/baseline_40"
            # lora_rank = 6
            
    time_delay_to_avoid_deadlock = 30 #seconds
    print(f"Waiting {time_delay_to_avoid_deadlock} seconds before starting evaluation.")
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    time.sleep(time_delay_to_avoid_deadlock)
    print(f"Model was trained and saved at {model_and_tokenizer_save_path}.")
    # return training_outputs

def evaluate_parallelized(
    model_name,
    saved_model_path,
    saved_tokenizer_path,
    results_filename,
    evaluation_type,
    data_path,
    lora_rank,
    test_random_state
):
    # start time
    import pytz
    import itertools
    import multiprocessing
    from datetime import datetime
    from functools import partial
    
    central_tz = pytz.timezone("America/Chicago")
    start_time = datetime.now(central_tz)#.strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # Create a pool of worker processes
    # Assign tasks to GPUs in a round-robin manner
    # task and gpu pairs
    tasks = list(DATASETS.keys())
    # gpu_ids = [0, 1, 2, 0, 1, 2, 0, 1]#[None]*len(tasks)
    gpu_ids = [0]*len(tasks)#[2]*2+[3]*6#+[2]*2
    
    # tasks = tasks[:1]
    # gpu_ids = gpu_ids[:1]
    print(gpu_ids)
    task_gpu_pairs = zip(tasks, itertools.cycle(gpu_ids))  # Cycles through GPUs
    
    num_workers = min(len(tasks), len(gpu_ids))  # Limit workers to available GPUs or tasks
    
    # Launch processes
    print(f"Evaluating Model: {model_name} | Evaluation Data Path: {data_path}")
    partial_evaluate_func = partial(
        evaluate, model_name=model_name, saved_model_path=saved_model_path, 
        saved_tokenizer_path=saved_tokenizer_path, results_filename=results_filename,
        evaluation_type=evaluation_type, data_path=data_path,
        lora_rank=lora_rank, test_random_state=test_random_state, evaluate_tot=False, evaluate_direct=True, evaluate_self_check=True)
    
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.starmap(partial_evaluate_func, task_gpu_pairs)
        # results = [pool.apply_async(partial_evaluate_func, (task, gpu)) for task, gpu in task_gpu_pairs]
        # results = [r.get() for r in results]  # Retrieve results
        # # Close the pool
        # pool.close()
        # pool.join()
    
    # end time
    end_time = datetime.now(central_tz)#.strftime("%Y-%m-%d %H:%M:%S UTC")
    total_time = end_time - start_time
    print(f"Total time take: {total_time}")
    print(f"Total minutes: {total_time.total_seconds()/60.0}")

def baseline_evaluate(training_outputs, results_filename="evaluation.json", evaluation_type="baseline", data_path="data/thought_data/llama/"):
    # evaluate_parallelized()
    for training_output in training_outputs:
        model_name = training_output[0]
        model_and_tokenizer_save_path = training_output[1]
        lora_rank = training_output[2]
        random_state = training_output[3]
        print(f"Entering baseline evaluation. Model Name: {model_name} | Model Path: {model_and_tokenizer_save_path} | Lora Rank: {lora_rank} | Random State: {random_state}")
        evaluate_parallelized(
            model_name=model_name,
            saved_model_path=model_and_tokenizer_save_path,
            saved_tokenizer_path=model_and_tokenizer_save_path,
            results_filename=results_filename,
            evaluation_type=evaluation_type,
            data_path=data_path,
            lora_rank=lora_rank,
            test_random_state=random_state
        )
            

# if __name__=="__main__":
#     baseline_train(
#         model=None,
#         model_name="meta-llama/Llama-3.1-8B-Instruct",
#         trained_model_save_path="src/saved_models/baseline/",
#         device="auto"
#     )