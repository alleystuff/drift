# import re
# import ast
# import json
import os
os.chdir("/aiau010_scratch/azm0269/federated_reasoning")
# os.chdir("/home/azm0269@auburn.edu/federated_reasoning")

import gc
import torch
import math
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from dotenv import load_dotenv
load_dotenv()

from datetime import datetime
from utils.utils import append_dict_to_json

from src.methods.selfcheck import self_check_evaluate
from src.methods.models import StateModel, TrainedModel
from src.prompts.prompt_templates import final_answers

from config_folder.model_config import MODELS
from config_folder.client_config_file import TEST_RANDOM_STATES
from utils.data_utils import get_train_test_split, get_task_data_dict

# def get_accuracy(prompt, model, model)
def get_accuracy_success_rate(task_name, prompt, model, model_parameters, tot=True, direct=True, **data_dict):
    """Measure Accuracy, Success Rate, and Self Check

    Args:
        task_name (_type_): name of the task
        prompt (_type_): prompt to get answer
        model (_type_): model to get answer
        model_parameters (_type_): model parameters
    """
    from thefuzz import fuzz
    # from nltk.tokenize import sent_tokenize
#     answer_accuracy_verification_prompt = '''Does {answer} match {final_answer}. 
# Respond with ONLY "yes" if {answer} matches {final_answer}. Respond with ONLY "no" if {answer} does not match {final_answer}.'''
    tot_answers = list()
    direct_answers = list()
    tot_verification = list()
    direct_verification = list()
    print(f"Evaluating a question from: {task_name}")
    for _ in range(4):
        if tot:
            tot_answer = model.generate(
                prompt=prompt,
                parameters=model_parameters
            )
            tot_answers.append(tot_answer)
            verify_answer_tot = fuzz.partial_ratio(str(data_dict['data_dict']['final_answer']), tot_answer)/100
            tot_verification.append(verify_answer_tot)
        # print(f"Tot answer: {tot_answer}")
        # print(str(data_dict['data_dict']['answer_options']))
        
        if direct:
            direct_answer_prompt = f"""Question: {data_dict['data_dict']['question']}. Provide an answer followed by your explanation. Pick an answer from these options: {data_dict['data_dict']['answer_options']}."""
            # print(direct_answer_prompt)
            direct_answer = model.generate(
                prompt=direct_answer_prompt,
                parameters=model_parameters
            )
            direct_answers.append(direct_answer)
            # print(f"Direct answer: {direct_answer}")
            verify_answer_direct = fuzz.partial_ratio(str(data_dict['data_dict']['final_answer']), direct_answer)/100
            # print(verify_answer_tot, verify_answer_direct)
            direct_verification.append(verify_answer_direct)
        # task_accuracy_verification_prompt = answer_accuracy_verification_prompt.format(answer=answer, final_answer=data_dict)
        # verify_answer = model.generate(
        #     prompt=task_accuracy_verification_prompt,
        #     parameters=model_parameters
        # ).lower()
        # verification.append(verify_answer)
        
    # tot accuracy
    if tot:
        tot_binary_accuracy = [1 if v > 0.75 else 0 for v in tot_verification]
        tot_accuracy = 1 if sum(tot_binary_accuracy)/len(tot_answers) > 0.5 else 0
    else:
        tot_binary_accuracy = None
        tot_accuracy = None
    
    # direct accuracy
    if direct:
        direct_binary_accuracy = [1 if v > 0.75 else 0 for v in direct_verification]
        direct_accuracy = 1 if sum(direct_binary_accuracy)/len(direct_answers) > 0.5 else 0
    else:
        direct_binary_accuracy = None
        direct_accuracy = None

    # print(tot_answers)
    # print(direct_answers)
    # success rate
    tot_success_rate = 0
    direct_success_rate = 0
    if task_name=="cose" or task_name=="csqa" or task_name=="medqa" or task_name=="piqa":
        # measure tot success rate
        if tot:
            for thought in data_dict['data_dict']['thought_sentences']:
                tot_success_rate += fuzz.partial_ratio(
                    thought, 
                    data_dict['data_dict']['final_answer'])/(100*len(data_dict['data_dict']['thought_sentences']))
        else:
            tot_success_rate = None
        
        # measure direct success rate   
        if direct:
            for direct_answer in direct_answers:
                direct_success_rate += fuzz.partial_ratio(
                    direct_answer, 
                    data_dict['data_dict']['final_answer'])/(100*len(direct_answers))
                # print(direct_success_rate)
        else:
            direct_success_rate = None
            
    elif task_name=="aqua" or task_name=="mathqa" or task_name=="medmcqa" or task_name=="pubmedqa":
        # check if the rational given in the dataset is a nan - happens in medmcqa
        try:
            rationale = data_dict['data_dict']['final_answer'] if math.isnan(data_dict['data_dict']['rationale']) else data_dict['data_dict']['rationale']
        except TypeError:
            rationale = data_dict['data_dict']['rationale']
        
        if tot:
            for thought in data_dict['data_dict']['thought_sentences']:
                tot_success_rate += fuzz.partial_ratio(
                    thought, 
                    rationale)/(100*len(data_dict['data_dict']['thought_sentences']))
                # print(thought)
                # print(rationale)
                # print("*"*100)
        else:
            tot_success_rate = None
        
        if direct:
            for direct_answer in direct_answers:
                direct_success_rate += fuzz.partial_ratio(
                    direct_answer, 
                    rationale)/(100*len(direct_answers))
        else:
            direct_success_rate = None
        
    return tot_accuracy, direct_accuracy, tot_success_rate, direct_success_rate

def evaluate(
    task_name,
    gpu_id,
    model=None,
    tokenizer=None,
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    saved_model_path=None,
    saved_tokenizer_path=None,
    results_filename="evaluation_results.json",
    evaluation_type="baseline",
    data_path = "data/thought_data/llama/", #this is where the raw data lives
    lora_rank=8,
    test_random_state=40,
    evaluate_tot=True,
    evaluate_direct=True,
    evaluate_self_check=True,
    client_id=None,
    server_strategy=None,
    server_round=None
):
    """This function will run evaluation on all tasks and compute accuracy and success rate.

    Args:
        results_filename (str, optional): where metrics will be stored. Defaults to "data/evaluation/baseline/evaluation_results.json".
    """
    if evaluate_tot:
        print("Evaluating ToT")
    if evaluate_direct:
        print("Evaluating Direct")
    if evaluate_self_check:
        print("Evaluating Self Check")
    
    if saved_model_path and saved_tokenizer_path:
        model = TrainedModel(
            base_model_name=model_name,
            saved_model_path=saved_model_path,
            tokenizer_path=saved_tokenizer_path,
            gpu_id=gpu_id,
            quantization="4bit"
        )
        print(f"Locally trained model initialized from {saved_model_path}. Model name: {model_name}")
    elif model!=None and tokenizer!=None:
        model = TrainedModel(
            base_model_name=model_name,
            saved_model_path=saved_model_path,
            tokenizer_path=saved_tokenizer_path,
            model=model,
            tokenizer=tokenizer,
            gpu_id=gpu_id,
            quantization="4bit"
        )
        print(f"Initialized model from huggingface. Model name: {model_name}")
    
    print(f"Using model: {model_name}")
    model_parameters = MODELS[model_name]['parameters']
    friendly_model_name = MODELS[model_name]['model_name']

    results_filename = f"data/evaluation/{evaluation_type}/{results_filename}"
    print(f"Results will be saved at: {results_filename}")
    
    # tasks = ['csqa', 'cose', 'aqua', 'mathqa', 'medmcqa', 'medqa', 'piqa', 'pubmedqa'] #list(DATASETS.keys())
    test_sample_size = 0.2 # 20% data used for testing
    # for task_name in tasks:
        # for test_random_state in TEST_RANDOM_STATES:
    tot_total_acc = 0
    tot_total_success_rate = 0
    direct_total_acc = 0
    direct_total_success_rate = 0
    
    #load thought data
    train_samples, test_samples = get_train_test_split(
        task_name = task_name,
        data_path = data_path,
        test_sample_size = test_sample_size, # 20% data used for testing
        test_random_state = test_random_state
    )
    # test_samples = test_samples.head(1)
    for i, row in test_samples.iterrows():
        # Convert string to dictionary using ast.literal_eval (safe parsing)
        if task_name=="csqa":
            data_dict = get_task_data_dict(row, task_name="csqa")
            prompt = final_answers[task_name]['final_answer_prompt_template']  
            answer_prompt=prompt.format(**data_dict)
            tot_accuracy, direct_accuracy, tot_success_rate, direct_success_rate = get_accuracy_success_rate(
                task_name=task_name,
                prompt=answer_prompt,
                model=model,
                model_parameters=model_parameters,
                data_dict=data_dict,
                tot=evaluate_tot,
                direct=evaluate_direct
            )
            # tot_total_acc, direct_total_acc, tot_total_success_rate, direct_total_success_rate = \
            #     tot_total_acc + tot_accuracy, \
            #     direct_total_acc + direct_accuracy, \
            #     tot_total_success_rate + tot_success_rate, \
            #     direct_total_success_rate + direct_success_rate
                
        if task_name=="cose":
            data_dict = get_task_data_dict(row, task_name="cose")        
            prompt = final_answers[task_name]['final_answer_prompt_template']
            answer_prompt=prompt.format(**data_dict)
            tot_accuracy, direct_accuracy, tot_success_rate, direct_success_rate = get_accuracy_success_rate(
                task_name=task_name,
                prompt=answer_prompt,
                model=model,
                model_parameters=model_parameters,
                data_dict=data_dict,
                tot=evaluate_tot,
                direct=evaluate_direct
            )
            # tot_total_acc, direct_total_acc, tot_total_success_rate, direct_total_success_rate = \
            #     tot_total_acc + tot_accuracy, \
            #     direct_total_acc + direct_accuracy, \
            #     tot_total_success_rate + tot_success_rate, \
            #     direct_total_success_rate + direct_success_rate
        
        if task_name=="aqua":
            data_dict = get_task_data_dict(row, task_name="aqua")        
            prompt = final_answers[task_name]['final_answer_prompt_template']
            answer_prompt=prompt.format(**data_dict)
            tot_accuracy, direct_accuracy, tot_success_rate, direct_success_rate = get_accuracy_success_rate(
                task_name=task_name,
                prompt=answer_prompt,
                model=model,
                model_parameters=model_parameters,
                data_dict=data_dict,
                tot=evaluate_tot,
                direct=evaluate_direct,
            )
            # tot_total_acc, direct_total_acc, tot_total_success_rate, direct_total_success_rate = \
            #     tot_total_acc + tot_accuracy, \
            #     direct_total_acc + direct_accuracy, \
            #     tot_total_success_rate + tot_success_rate, \
            #     direct_total_success_rate + direct_success_rate
            
        if task_name=="mathqa":
            data_dict = get_task_data_dict(row, task_name="mathqa")        
            prompt = final_answers[task_name]['final_answer_prompt_template']
            answer_prompt=prompt.format(**data_dict)
            tot_accuracy, direct_accuracy, tot_success_rate, direct_success_rate = get_accuracy_success_rate(
                task_name=task_name,
                prompt=answer_prompt,
                model=model,
                model_parameters=model_parameters,
                data_dict=data_dict,
                tot=evaluate_tot,
                direct=evaluate_direct,
            )
            # tot_total_acc, direct_total_acc, tot_total_success_rate, direct_total_success_rate = \
            #     tot_total_acc + tot_accuracy, \
            #     direct_total_acc + direct_accuracy, \
            #     tot_total_success_rate + tot_success_rate, \
            #     direct_total_success_rate + direct_success_rate
        
        if task_name=="medmcqa":
            data_dict = get_task_data_dict(row, task_name="medmcqa")        
            prompt = final_answers[task_name]['final_answer_prompt_template']
            answer_prompt=prompt.format(**data_dict)
            tot_accuracy, direct_accuracy, tot_success_rate, direct_success_rate = get_accuracy_success_rate(
                task_name=task_name,
                prompt=answer_prompt,
                model=model,
                model_parameters=model_parameters,
                data_dict=data_dict,
                tot=evaluate_tot,
                direct=evaluate_direct,
            )
            # tot_total_acc, direct_total_acc, tot_total_success_rate, direct_total_success_rate = \
            #     tot_total_acc + tot_accuracy, \
            #     direct_total_acc + direct_accuracy, \
            #     tot_total_success_rate + tot_success_rate, \
            #     direct_total_success_rate + direct_success_rate
        
        if task_name=="medqa":
            data_dict = get_task_data_dict(row, task_name="medqa")
            prompt = final_answers[task_name]['final_answer_prompt_template']
            answer_prompt=prompt.format(**data_dict)
            tot_accuracy, direct_accuracy, tot_success_rate, direct_success_rate = get_accuracy_success_rate(
                task_name=task_name,
                prompt=answer_prompt,
                model=model,
                model_parameters=model_parameters,
                data_dict=data_dict,
                tot=evaluate_tot,
                direct=evaluate_direct,
            )
            # tot_total_acc, direct_total_acc, tot_total_success_rate, direct_total_success_rate = \
            #     tot_total_acc + tot_accuracy, \
            #     direct_total_acc + direct_accuracy, \
            #     tot_total_success_rate + tot_success_rate, \
            #     direct_total_success_rate + direct_success_rate
        
        if task_name=="piqa":
            data_dict = get_task_data_dict(row, task_name="piqa")        
            prompt = final_answers[task_name]['final_answer_prompt_template']
            answer_prompt=prompt.format(**data_dict)
            tot_accuracy, direct_accuracy, tot_success_rate, direct_success_rate = get_accuracy_success_rate(
                task_name=task_name,
                prompt=answer_prompt,
                model=model,
                model_parameters=model_parameters,
                data_dict=data_dict,
                tot=evaluate_tot,
                direct=evaluate_direct,
            )
            # tot_total_acc, direct_total_acc, tot_total_success_rate, direct_total_success_rate = \
            #     tot_total_acc + tot_accuracy, \
            #     direct_total_acc + direct_accuracy, \
            #     tot_total_success_rate + tot_success_rate, \
            #     direct_total_success_rate + direct_success_rate
        
        if task_name=="pubmedqa":
            data_dict = get_task_data_dict(row, task_name="pubmedqa")        
            prompt = final_answers[task_name]['final_answer_prompt_template']
            answer_prompt=prompt.format(**data_dict)
            tot_accuracy, direct_accuracy, tot_success_rate, direct_success_rate = get_accuracy_success_rate(
                task_name=task_name,
                prompt=answer_prompt,
                model=model,
                model_parameters=model_parameters,
                data_dict=data_dict,
                tot=evaluate_tot,
                direct=evaluate_direct,
            )
            
        # aggregate
        if evaluate_tot: 
            tot_total_acc, tot_total_success_rate = \
                tot_total_acc + tot_accuracy, \
                tot_total_success_rate + tot_success_rate
        else:
            tot_total_acc, tot_total_success_rate = tot_accuracy, tot_success_rate

        if evaluate_direct:
            direct_total_acc, direct_total_success_rate = \
                direct_total_acc + direct_accuracy, \
                direct_total_success_rate + direct_success_rate
    
    # normalize for all samples to get average
    if evaluate_tot:
        tot_final_accuracy = tot_total_acc/len(test_samples)
        tot_final_success_rate = tot_total_success_rate/len(test_samples)
    else:
        tot_final_accuracy = None
        tot_final_success_rate = None

    if evaluate_direct:    
        direct_final_accuracy = direct_total_acc/len(test_samples)
        direct_final_success_rate = direct_total_success_rate/len(test_samples)
    else:
        direct_final_accuracy = None
        direct_final_success_rate = None
    # do self check
    # clean up the existing model to not overflow in memory
    # del model
    # gc.collect()
    # torch.cuda.empty_cache()
    
    if evaluate_self_check:
        try:
            self_check_confidence = self_check_evaluate(
                model_name,
                model=model,
                task_name=task_name,
                saved_model_path=saved_model_path,
                saved_tokenizer_path=saved_tokenizer_path,
                data_path=data_path,
                random_state=test_random_state
            )
        except Exception as e:
            print(e)
            self_check_confidence = str(e)
    else:
        self_check_confidence = None
    
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # add in client dict for metrics
    if evaluation_type=="baseline":
        metrics = {
            "task": task_name,
            "tot_accuracy": tot_final_accuracy,
            "tot_success_rate": tot_final_success_rate,
            "direct_accuracy": direct_final_accuracy,
            "direct_success_rate": direct_final_success_rate,
            "self_check_confidence": self_check_confidence,
            "datetime_saved": current_datetime,
            "evaluation_type": evaluation_type,
            "model_name": friendly_model_name,
            "lora_rank": lora_rank,
            "random_state": test_random_state,
        }
    elif evaluation_type=="federated":
        metrics = {
            "task": task_name,
            "tot_accuracy": tot_final_accuracy,
            "tot_success_rate": tot_final_success_rate,
            "direct_accuracy": direct_final_accuracy,
            "direct_success_rate": direct_final_success_rate,
            "self_check_confidence": self_check_confidence,
            "datetime_saved": current_datetime,
            "evaluation_type": evaluation_type,
            "model_name": friendly_model_name,
            "lora_rank": lora_rank,
            "client_id": client_id,
            "server_strategy": server_strategy,
            "server_round": server_round,
            "random_state": test_random_state,
        }
        
    # all_tasks_evaluation.append(metrics)
    append_dict_to_json(file_path=results_filename, new_data=metrics)
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
# if __name__=="__main__":
#     evaluate()