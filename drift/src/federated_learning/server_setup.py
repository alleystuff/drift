import os
import gc
import torch
import numpy as np
import pandas as pd
from torch import nn
import flwr as fl
# from flwr.common import (
#     FitRes,
#     Parameters,
#     Scalar
# )
from flwr.common import Context
from flwr.client import ClientApp
from flwr.simulation import run_simulation
# from flwr.server.client_proxy import ClientProxy
# from flwr.common.parameter import ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig, ServerAppComponents

from collections import OrderedDict
from typing import List, Tuple, Dict, Union, Optional

#data imports

#client imports

#model imports
# from src.models.server_models import initialize_model
from src.federated_learning.server import (fedavg, fedprox, fedadam, fedtrimmedavg, drift)
from src.federated_learning.client import CustomClient
from src.methods.models import ServerModel, TrainedModel

#config and util imports
from src.methods.dpo import get_dpo_trainer
from transformers import AutoModelForCausalLM, AutoProcessor
from config_folder import client_config_file#, server_config_file
# from config_folder.client_config_file import get_server_checkpoint_path
from utils.data_utils import get_train_test_split, get_preference_dataset
from utils.training_utils import get_quantization_config, get_peft_parameters_and_peft_state_dict, set_peft_state_dict

import random
# import warnings
# warnings.simplefilter('ignore')
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# torch.manual_seed(42)

def run_server(
    num_cpus,
    num_gpus,
    gpu_id=None,
    batch_size=4,
    lora_rank=8,
    server_rounds=500,
    quantization="4bit",
    strategy_name="FedAvg",
    thought_data_path="data/thought_data/llama/",
    model_name = "meta-llama/Llama-3.1-8B-Instruct",
    working_directory="/aiau010_scratch/azm0269/federated_reasoning",
    random_state=42
):
    """Run server"""
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)
    
    tasks = list(client_config_file.DATASETS.keys())
    print(f"Tasks: {tasks}")

    # load base model and tokenizer
    quantization_config = get_quantization_config(quantization=quantization)
    
    #if a saved model and tokenizer for the current strategy is present then load it from the saved directory else initialize a new model
    ## read the model from src/saved_models/server/{strategy}
    saved_model_path = f"src/saved_models/{strategy_name}"
    base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto", # Map the entire model to the specified GPU
            torch_dtype=torch.float16,  # Optionally use mixed precision
            quantization_config=quantization_config,  # Pass the BitsAndBytesConfig
    )
    tokenizer = AutoProcessor.from_pretrained(model_name)
    
    ##### initialize the dpo trainer once here and then delete it. This way you can also pull out the initial peft parameters and pass to the strategy as initial params
    ##### then in each FL round merge the update parameters from the server into the dpo_trainer.model
    ##### initialize and pass the tokenizer from here as well.
    train_dataset, test_dataset = get_train_test_split(
        task_name="cose",
        data_path=thought_data_path,
        test_sample_size=0.2,
        test_random_state=random_state
    )
    train_preference_dataset, test_preference_dataset = get_preference_dataset(train_dataset), get_preference_dataset(test_dataset)
    dpo_trainer = get_dpo_trainer(
        model = base_model,
        tokenizer=tokenizer,
        train_dataset=train_preference_dataset,
        test_dataset=test_preference_dataset,
        batch_size=4,
        output_dir="src/saved_models/clients/",
        run_name="federated",
    )
    # the flower server will call this strategy and this strategy will send the initial parameters below to the server.
    peft_parameters, peft_state_dict = get_peft_parameters_and_peft_state_dict(
        peft_model=dpo_trainer.model, 
        adpater_name='default'
    )
    initial_parameters = fl.common.ndarrays_to_parameters(peft_parameters) #set intial parameters
    
    # clean up
    del dpo_trainer, train_dataset, test_dataset, train_preference_dataset, test_preference_dataset, peft_parameters, peft_state_dict
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"Experiment Type: {strategy_name} | Random State: {random_state} | Total Server Rounds: {server_rounds}")
    def client_fn(context: Context):
        """
        Create a Flower client representing a single client.
        cid: client id - default argument expected by Flower and used by Flower during federated learning to pass index of the client to be selected
        """
        # os.chdir(working_directory)
        # print(os.getcwd())

        cid = context.node_config["partition-id"]
        total_clients = context.node_config["num-partitions"]
        
        # pop one task at a time for each of the clients and initialize the preference data
        # pass this data into the CustomClient class.
        # current_task = tasks.pop()
        current_task = tasks[int(cid)]
        
        ## if you run experiments for more than 8 clients then randomly pick an integer between 0-7 to index on tasks
        ## and assign a dataset to a client. This will create a both homogenous and heterogenous set up
        
        train_dataset, test_dataset = get_train_test_split(
            task_name=current_task,
            data_path=thought_data_path,
            test_sample_size=0.2,
            test_random_state=random_state
        )
        train_preference_dataset, test_preference_dataset = get_preference_dataset(train_dataset), get_preference_dataset(test_dataset)
        client_model_save_path = f"src/saved_models/clients/client_{cid}"
        dpo_trainer = get_dpo_trainer(
            model = base_model,
            tokenizer=tokenizer,
            train_dataset=train_preference_dataset,
            test_dataset=test_preference_dataset,
            batch_size=batch_size,
            output_dir=client_model_save_path,
            lora_rank=lora_rank,
            run_name="federated",
            fl_rounds=server_rounds
        )
        print(f"Client ID: {cid} | Client Task Name: {current_task} | Client Model Save Path: {client_model_save_path} | Total Clients: {total_clients}")

        ### if no tasks left in the task list then return nothing else a CustomClient
        # if len(tasks)==0:
        #     return
        # else:
        return CustomClient(    
            dpo_trainer=dpo_trainer,
            task_name=current_task,
            model_name=model_name,
            train_size=len(train_dataset),
            client_id=str(cid),
            data_path=thought_data_path,
            lora_rank=lora_rank,
            random_state=random_state,            
        ).to_client()

    def fit_config(server_round: int):
        """
        Config function used to pass variables and data to the local clients during federated learning
        The federated learning strategy will call this function every round.
        """
        config = {
            "server_round": server_round,
            "strategy": strategy_name
        }
        return config
    
    def server_fn(context: Context):
        if strategy_name=="FedAvg":
            strategy = fedavg.fedavg_strategy(
                initial_parameters=initial_parameters,
                client_config=fit_config,
                strategy_name=strategy_name,
            )
        if strategy_name=="FedProx":
            strategy = fedprox.fedprox_strategy(
                initial_parameters=initial_parameters,
                client_config=fit_config,
                strategy_name=strategy_name,
            )
        if strategy_name=="FedAdam":
            strategy = fedadam.fedadam_strategy(
                initial_parameters=initial_parameters,
                client_config=fit_config,
                strategy_name=strategy_name,
            )
        if strategy_name=="FedTrimmedAvg":
            strategy = fedtrimmedavg.fedtrimmedavg_strategy(
                initial_parameters=initial_parameters,
                client_config=fit_config,
                strategy_name=strategy_name,
            )
        if strategy_name=="Drift":
            strategy = drift.drift_strategy(
                initial_parameters=initial_parameters,
                client_config=fit_config,
                strategy_name=strategy_name,
            )
        server_config = ServerConfig(
            num_rounds = server_rounds
        )
        return ServerAppComponents(strategy=strategy, config=server_config)
    
    ##define client resources and start simulations based on the current strategy
    backend_config = {
        "client_resources": {
            "num_cpus": num_cpus, 
            "num_gpus": num_gpus
        }
    } 

    server = ServerApp(
        server_fn=server_fn
    )

    client = ClientApp(
        client_fn=client_fn
    )
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=client_config_file.NUM_CLIENTS,
        backend_config=backend_config
    )

    return f"Completed simulation/experiment: {strategy_name} | Strategy: {strategy_name} | Random State: {random_state}"