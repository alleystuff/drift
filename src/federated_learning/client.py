import os
import csv
import json
import torch
import numpy as np
import flwr as fl
import pandas as pd
from peft import LoraConfig, PeftModel
from collections import OrderedDict
from config_folder import client_config_file
from src.methods.dpo import get_dpo_trainer
from src.methods.evaluate import evaluate
from utils.data_utils import get_train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.training_utils import (
    print_trainable_parameters, get_peft_parameters_and_peft_state_dict, 
    set_peft_state_dict, get_quantization_config
)
from utils.utils import append_dict_to_json


class CustomClient(fl.client.NumPyClient):
    def __init__(self, dpo_trainer, task_name, model_name, train_size, client_id, data_path, lora_rank=8, gpu_id=None, random_state=42):
        self.dpo_trainer = dpo_trainer
        self.task_name=  task_name
        self.model_name = model_name
        self.train_size = train_size
        self.client_id = client_id
        self.data_path = data_path
        self.lora_rank = lora_rank
        self.device_map = gpu_id if gpu_id!=None and type(gpu_id)==int else "auto"
        self.random_state = random_state
        
    def set_parameters(self, parameters):
        """
        - Replace the parameters of the adapter with the parameters from the server.
        """
        params, state_dict = get_peft_parameters_and_peft_state_dict(self.dpo_trainer.model)
        state_dict_keys = list(state_dict.keys())
        state_dict_values = list(state_dict.values())

        updated_state_dict = OrderedDict()
        for p, k, v in zip(parameters, state_dict_keys, state_dict_values):
            # torch_data_type, torch_device, = v.dtype, f"{v.device}:{v.device.index}"
            updated_state_dict[k] = v[:] = torch.from_numpy(p)
        _ = set_peft_state_dict(self.dpo_trainer.model, updated_state_dict, adapter_name="default")
        
    def get_parameters(self):
        # check to see the parameters actually got merged in
        params_after_update, state_dict_after_update = get_peft_parameters_and_peft_state_dict(self.dpo_trainer.model)
        return params_after_update
      
    def fit(self, parameters, config):
        """This is where the aggregated parameters will merge with the peft model and local training will happen.
        1. Load preference data using get_train_test_split. Use task provided by the config. The config is passed by the server.
        2. Use the self.base_model as the model on which the aggeregated parameters will be merged and merge the model.
        3. Use the get_dpo_trainer to get the trainer for the merged model.
        4. Train the model
        5. Pull the trained parameters from the trained model using get_parameters and get_peft_parameters_and_peft_state_dict.
        6. Every 5 server rounds do model evaluation using evaluate_and_save_metrics.
        Args:
            parameters (_type_): _description_
            config (_type_): _description_
        """
        #config utilized by flower during federation
        self.server_round = config["server_round"]
        self.strategy = config["strategy"]
        
        
        
        #set the adapter parameters received from the server
        self.set_parameters(parameters=parameters)
        
        # one of the clients will save the model for the respective strategy in the respective strategy's folder
        all_server_strategies = ['FedAvg', 'FedProx']
        if int(self.server_round)%5==0 and self.strategy=='FedAvg' and int(self.client_id)==1:
            server_model_directory = f"src/saved_models/server/{self.strategy}/"
            self.dpo_trainer.save_model(server_model_directory) #save the trained client model
        
        # if its our strategy then the model will be saved under the client folder for all clients
        if int(self.server_round)%5==0 and self.strategy=='Drift':
            self.dpo_trainer.save_model() 
        
        # maintain training logs through only one of the clients to avoid duplicate entries
        if int(self.client_id)==1:
            training_log = {
                'strategy': self.strategy,
                'server_round': self.server_round,
                'seed': self.random_state,
                'lora_rank': self.lora_rank
            }
            training_log_filename = '/training_log.json'
            append_dict_to_json(
                file_path=training_log_filename,
                new_data=training_log
            )
        
        # self.evaluate_and_save_metrics() # evaluate and save metrics from the aggregated parameters
        
        self.dpo_trainer.train() #train the model
        
        
        
        return self.get_parameters(), self.train_size, {}
    
    def evaluate_and_save_metrics(self):
        evaluate(
            task_name=self.task_name,
            gpu_id=None,
            model=self.dpo_trainer.model,
            tokenizer=self.dpo_trainer.tokenizer,
            model_name=self.model_name,
            saved_model_path=None,
            saved_tokenizer_path=None,
            results_filename="evaluation_results.json",
            evaluation_type="federated",
            data_path = self.data_path, #this is where the raw data lives
            lora_rank=self.lora_rank,
            test_random_state=self.random_state,
            evaluate_tot=False,
            evaluate_direct=True,
            evaluate_self_check=True,
            client_id=self.client_id,
            server_strategy=self.strategy,
            server_round=self.server_round
        )
