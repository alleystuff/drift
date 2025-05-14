import torch
import flwr as fl
import numpy as np
from flwr.common import (
    FitRes,
    Parameters,
    Scalar
)
from collections import OrderedDict
from flwr.server.client_proxy import ClientProxy
from typing import List, Tuple, Dict, Union, Optional
from config_folder import client_config_file
from config_folder.client_config_file import get_server_checkpoint_path


def fedavg_strategy(initial_parameters, client_config, strategy_name, adpater_name="default", seed=42):
    """
    Implement FedAvg Strategy
    """
    class FedAvgModelStrategy(fl.server.strategy.FedAvg):
        def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            """Aggregate model weights using weighted average and store checkpoint"""
            # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics - returns aggregated parameters and aggregated_metrics
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

            if aggregated_parameters is not None:
                # Convert `Parameters` to `List[np.ndarray]`
                aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

                # Convert `List[np.ndarray]` to PyTorch`state_dict`
                # update the peft state dict with aggregated parameters
                # updated_peft_state_dict = zip(peft_state_dict.state_dict().keys(), aggregated_ndarrays)
                # updated_peft_state_dict = OrderedDict({k: torch.tensor(v) for k, v in updated_peft_state_dict})
                
                # set the peft model's state dict with the updated peft state dict which contains the aggregated parameters
                # peft_model.load_state_dict(updated_peft_state_dict, strict=True)
                # model = set_peft_state_dict(
                #     peft_model=peft_model,
                #     peft_state_dict=peft_state_dict,
                #     adapter_name=adpater_name
                # )

                # # Save the model
                # server_checkpoint_path = get_server_checkpoint_path(
                #     filename=f"{strategy_name}_{seed}",
                #     model_type="server"
                # )
                # model.save_pretrained(server_checkpoint_path)

            return aggregated_parameters, aggregated_metrics

    strategy = FedAvgModelStrategy(
        fraction_fit = 0.8,
        fraction_evaluate = 1.0,
        min_fit_clients = int(0.8*client_config_file.NUM_CLIENTS),
        min_evaluate_clients = client_config_file.NUM_CLIENTS,
        min_available_clients = client_config_file.NUM_CLIENTS,
        on_fit_config_fn = client_config,
        on_evaluate_config_fn= client_config,
        initial_parameters=initial_parameters
    )
    return strategy