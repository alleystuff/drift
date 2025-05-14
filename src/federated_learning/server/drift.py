import torch
import flwr as fl
import numpy as np
import networkx as nx
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from collections import OrderedDict
from flwr.server.client_proxy import ClientProxy
from typing import List, Tuple, Dict, Union, Optional
from config_folder import client_config_file
from config_folder.client_config_file import get_server_checkpoint_path


def drift_strategy(initial_parameters, client_config, strategy_name, adpater_name="default", seed=42):
    """
    Implement FedAvg Strategy
    """
    class DriftModelStrategy(fl.server.strategy.FedAvg):
        
        def configure_fit(self, server_round, parameters, client_manager):
            #if its round 1 then only initial parameters are sent to the clients
            if server_round==1:
                return super().configure_fit(server_round, parameters, client_manager)
            if server_round>1:
                for client_proxy, fit_res in self.results:
                    print(client_proxy.cid, len(parameters_to_ndarrays(fit_res.parameters)))
                    # all logic for drift goes here
                
                    
        def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            """Aggregate model weights using weighted average and store checkpoint"""
            # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics - returns aggregated parameters and aggregated_metrics
            self.results = results # these are all FitRes which will be used on configure_fit
            
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

            if aggregated_parameters is not None:
                # Convert `Parameters` to `List[np.ndarray]`
                aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

                """
                update FitRes here and use it inside configure_fit
                apply drift here
                - save client graph data here as well 
                - link client id to each clients parameters using fl.server.client_proxy.ClientProxy from results
                - update self.all_fit_ins here and use it inside configure_fit
                
                another and maybe better option is to just do all aggregation for each client inside configure_fit using the fit_res which can be saved from here into a class attribute
                """

            return aggregated_parameters, aggregated_metrics

    strategy = DriftModelStrategy(
        fraction_fit = 0.3,
        fraction_evaluate = 0,
        min_fit_clients = int(0.3*client_config_file.NUM_CLIENTS),
        min_evaluate_clients = 0,
        min_available_clients = client_config_file.NUM_CLIENTS,
        on_fit_config_fn = client_config,
        on_evaluate_config_fn= client_config,
        initial_parameters=initial_parameters
    )
    return strategy