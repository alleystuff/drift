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


def fedprox_strategy(initial_parameters, client_config, strategy_name, proximal_mu, adpater_name="default", seed=42):
    """
    Implement FedAvg Strategy
    """
    class FedProxModelStrategy(fl.server.strategy.FedProx):
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

            return aggregated_parameters, aggregated_metrics

    strategy = FedProxModelStrategy(
        fraction_fit = 0.8,
        fraction_evaluate = 1.0,
        min_fit_clients = int(0.8*client_config_file.NUM_CLIENTS),
        min_evaluate_clients = client_config_file.NUM_CLIENTS,
        min_available_clients = client_config_file.NUM_CLIENTS,
        on_fit_config_fn = client_config,
        on_evaluate_config_fn= client_config,
        initial_parameters=initial_parameters,
        proximal_mu=proximal_mu
    )
    return strategy