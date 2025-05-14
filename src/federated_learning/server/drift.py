import torch
import random
import flwr as fl
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
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
from collections import OrderedDict, defaultdict
from flwr.server.client_proxy import ClientProxy
from typing import List, Tuple, Dict, Union, Optional
from config_folder import client_config_file
from config_folder.client_config_file import get_server_checkpoint_path
from utils.training_utils import measure_symmetric_kl_divergence
from utils.utils import append_dict_to_json, append_graph_dict_to_json

########################### Drift Minimum Spanning Tree ###################################

def drift_mst_strategy(
    initial_parameters, client_config, friendly_model_name, fraction_fit=0.3, 
    strategy_name="Drift_MST", method="kruskal", adpater_name="default", seed=42
):
    """
    Implement FedAvg Strategy
    """
    
    def get_client_graph(all_fit_res, weight_name='kl_div'):
        """Generate and return a client graph

        Args:
            all_fit_res (_type_): list containing all fit results
        """
        client_graph = nx.Graph()
        kl_list = list()
        for client_proxy_i, fit_in in all_fit_res: #replace enumerate with client ID from client proxy
            for client_proxy_j, fit_in_neighbor_client in all_fit_res:
                if client_proxy_i.cid!=client_proxy_j.cid:
                    kl_list.append(
                        (
                            client_proxy_i.cid, client_proxy_j.cid, 
                            {weight_name: measure_symmetric_kl_divergence(
                                parameters_to_ndarrays(fit_in.parameters), 
                                parameters_to_ndarrays(fit_in_neighbor_client.parameters))
                            }
                        )
                    )

        #create graph for visual and search
        client_graph.add_edges_from(kl_list) 
        return client_graph
    
    ##add KL aggregate and aggregate_and_return_fitins
    def kl_aggregate(current_client_idx, weights, all_fit_res):
        """
        1. take all incoming weights
        2. add a weight of 1.0 for the current client to weights
        3. normalize all weights 0-1
        4. aggregate params for all clients using normalized weights
        5. take weighted average of the current client
        6. add 4 and 5
        """
        max_weight = np.float64(np.array([i[1] for i in weights]).max()) #pull the max weight and assign to the current client - the current client and most closest client will have the highest weight
        current_client_cid_and_weight = (all_fit_res[current_client_idx][0].cid, max_weight) #current client is assigned the highest weight 
        weights.append(current_client_cid_and_weight) #append weight for current client to all incoming weights
        print(f"Weights for the current round before normalization: {weights}")
        
        weights = weights[::-1] #reverse list
        all_weights_sum = sum([i[1] for i in weights]) #sum all weights to get total_sum for normalization - remember weight for the client is at index 1
        final_adjusted_weights = [(i[0], i[1]/all_weights_sum) for i in weights] #normalie all weights between 0-1 
        print(f"Weights for the current round after normalization: {final_adjusted_weights}")
        
        current_client_cid_and_weight_normalized = final_adjusted_weights[0] #get the weight for the current client
        # print(f"current client cid and weights adjusted: {current_client_cid_and_weight_normalized}")
        # client_fit_res_as_key_value = defaultdict()
        params = parameters_to_ndarrays(all_fit_res[current_client_idx][1].parameters) # fit_res is a list of tuples of [client_proxy, client_fit_res]
        params = [p*current_client_cid_and_weight_normalized[1] for p in params] #take weighted average of the current client
        # print(f"final adjusted weights: {final_adjusted_weights}")
        all_non_current_client_params = list()
        for i, (client_proxy, fit_res) in enumerate(all_fit_res):
            for client_weight in final_adjusted_weights[1:]: #loop over all client except for the current client which is at index 0
                # print(f"Client proxy weight inside loop: {client_proxy.cid} | Client weight inside loop: {client_weight}")
                if client_proxy.cid == client_weight[0]:
                    # print(f"Client cid: {client_weight[0]} | Client weight: {client_weight[1]}")
                    non_current_client_params = [
                        p * client_weight[1] 
                        for p in parameters_to_ndarrays(fit_res.parameters) # client_weight[0] is the name of the current client given as cid
                    ]
                    all_non_current_client_params.append(non_current_client_params)
        non_current_client_params = [np.sum(arrays, axis=0) for arrays in zip(*all_non_current_client_params)]
        aggregated_params = [current_client_param + non_current_client_param for current_client_param, non_current_client_param in zip(params, non_current_client_params)]
        return aggregated_params
    
    def aggregate_and_return_fitins(all_fit_res, client_specific_pairs, config={}, weight_name='kl_div'):
        aggregated_fit_ins = []
        for current_client_idx, (current_client_proxy, _) in enumerate(all_fit_res):
            current_client_specific_pairs = client_specific_pairs[current_client_proxy.cid] #pull the connected clients by the current client's cid
            
            #pull weights from the pairs which show a connection to the current client
            client_weights = list()
            for pair in current_client_specific_pairs:
                client_weights.append(pair[2][weight_name])
            
            
            #inverse and normalize weights - taking the inverse means that client with less KL divergence has more weight 
            client_weights = 1/(np.array(client_weights)+1e-6) #add a small constant to avoid divide by zero
            normalized_client_weights = client_weights/client_weights.sum()
            client_weights = list()
            for pair, w in zip(current_client_specific_pairs, normalized_client_weights):
                if pair[0]!=current_client_proxy.cid:
                    client_weights.append((pair[0], w))
                elif pair[1]!=current_client_proxy.cid:
                    client_weights.append((pair[1], w))
            
            current_client_agg_params = kl_aggregate(
                current_client_idx, 
                client_weights, 
                all_fit_res=all_fit_res
            )
            config["cid"] = current_client_proxy.cid
            config = dict(config)
            # print(f"Inside aggregate_and_return_fitins: Current Client Proxy CID: {current_client_proxy.cid}| Current client config dict: {config} | Len client agg param: {len(current_client_agg_params)}")
            aggregated_fit_ins.append(
                (
                    current_client_proxy.cid,
                    FitIns(
                        parameters=ndarrays_to_parameters(current_client_agg_params),
                        config=config
                    )
                )
            )
        return aggregated_fit_ins
    
    def aggregate_drift_mst(all_fit_res, server_round, weight='kl_div', method='kruskal', config={}):
        client_graph = get_client_graph(
            all_fit_res=all_fit_res,
            weight_name=weight
        ) #get client graph
        
        #create a minimum spanning tree
        mst = nx.minimum_spanning_tree(
            G=client_graph, 
            weight=weight, 
            algorithm=method
        )
        
        graph_data_dict = {
            "server_round": server_round,
            "server_strategy": strategy_name,
            "model_name": friendly_model_name,
            "complete_client_graph": json_graph.node_link_data(client_graph, edges='links'),
            "server_strategy_graph":  json_graph.node_link_data(mst, edges='links'),
            "random_state": seed
        }
        # print(graph_data_dict)
        append_graph_dict_to_json(
            filename="data/drift/drift_mst_graph_data.json",
            graph_data=graph_data_dict
        )
        
        # get edges of MST
        mst_edgs = mst.edges(data=True)
        client_specific_pairs = defaultdict(list)
        for edge in mst_edgs:
            for client_proxy, client_fit_res in all_fit_res:
                if client_proxy.cid==edge[0] or client_proxy.cid==edge[1]:
                    client_specific_pairs[client_proxy.cid].append(edge)

        # for i, j in all_fit_res:
        #     print(i.cid, type(j.parameters))
            
        aggregated_fit_ins = aggregate_and_return_fitins(
            all_fit_res=all_fit_res,
            client_specific_pairs=client_specific_pairs,
            config=config
        )
        
        return aggregated_fit_ins
        
    class DriftMSTModelStrategy(fl.server.strategy.FedAvg):
        
        def configure_fit(self, server_round, parameters, client_manager):
            #if its round 1 then only initial parameters are sent to the clients
            if server_round==1:
                return super().configure_fit(server_round, parameters, client_manager)
            if server_round>1:
                config = defaultdict()
                # if self.on_fit_config_fn is not None:
                config["server_round"] = server_round 
                config["strategy"]= strategy_name
                
                # clients which did not participate in the previous round get the standard aggregated parameters using fedavg strategy
                # Sample clients
                sample_size, min_num_clients = self.num_fit_clients(
                    client_manager.num_available()
                )
                # print(sample_size, min_num_clients)
                clients = client_manager.sample(
                    num_clients=sample_size, min_num_clients=min_num_clients
                )
                # print(f"Config before drift aggregation: {config}")
                aggregated_fit_ins = aggregate_drift_mst(
                    all_fit_res=self.all_fit_res,
                    server_round=server_round,
                    weight='kl_div',
                    method=method,
                    config=config
                )

                # Return client/config pairs
                configure_client_fitins = []
                # for client, fit_ins in zip(clients, aggregated_fit_ins):
                #     if int(client.cid)==int(fit_ins[0]):
                #         configure_client_fitins.append((client, FitIns(fit_ins[1].parameters, {"server_round": server_round, "strategy": "drift_mst", "cid": client.cid})))#fit_ins[1]))
                #     elif int(client.cid)!=int(fit_ins[0]):
                #         configure_client_fitins.append((client, FitIns(parameters, {"server_round": server_round, "strategy": "drift_mst", "cid": client.cid})))
                for client in clients:
                    for fit_ins in aggregated_fit_ins:
                        if int(client.cid)==int(fit_ins[0]):
                            configure_client_fitins.append((client, FitIns(fit_ins[1].parameters, {"server_round": server_round, "strategy": strategy_name, "cid": client.cid})))#fit_ins[1]))
                
                # print(f"Len of configure client fitins: {len(configure_client_fitins)}")
                # for client, fit_ins in configure_client_fitins:
                #     print(f"Sending - Sampled Client CID: {client.cid} | Fit in config: {fit_ins.config}")
        
                # print("*"*100)
                
                existing_clients = [int(c.cid) for c, _ in configure_client_fitins] #list of all the clients which have aggregated parameters - all the others get fedavg aggregated params
                for client in clients:
                    if int(client.cid) not in existing_clients:
                        configure_client_fitins.append((client, FitIns(parameters, {"server_round": server_round, "strategy": strategy_name, "cid": client.cid})))
                    
                
                print(f"Len of updated configure client fitins: {len(configure_client_fitins)}")
                for client, fit_ins in configure_client_fitins:
                    print(f"Sending - Sampled Client CID: {client.cid} | Fit in config: {fit_ins.config}")
        
                print("*"*100)
                
                return configure_client_fitins
                
                    
        def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            """Aggregate model weights using weighted average and store checkpoint"""
            # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics - returns aggregated parameters and aggregated_metrics
            self.all_fit_res = results # these are all FitRes which will be used on configure_fit
            
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

    strategy = DriftMSTModelStrategy(
        fraction_fit = fraction_fit,
        fraction_evaluate = 0,
        min_fit_clients = int(fraction_fit*client_config_file.NUM_CLIENTS),
        min_evaluate_clients = 0,
        min_available_clients = client_config_file.NUM_CLIENTS,
        on_fit_config_fn = client_config,
        on_evaluate_config_fn= client_config,
        initial_parameters=initial_parameters
    )
    return strategy

########################### Drift Shortest Path ###################################

def drift_sp_strategy(
    initial_parameters, client_config, friendly_model_name, fraction_fit=0.3, 
    weight='kl_div', method='dijkstra', divergence_threshold=0.5,
    strategy_name="drift_sp", adpater_name="default", seed=42
):
    """
    Implement Drift Shortest Path
    """
    def get_client_graph(all_fit_res, weight_name='kl_div'):
        """Generate and return a client graph

        Args:
            all_fit_res (_type_): list containing all fit results
        """
        client_graph = nx.Graph()
        kl_list = list()
        for client_proxy_i, fit_in in all_fit_res: #replace enumerate with client ID from client proxy
            for client_proxy_j, fit_in_neighbor_client in all_fit_res:
                if client_proxy_i.cid!=client_proxy_j.cid:
                    kl_list.append(
                        (
                            client_proxy_i.cid, client_proxy_j.cid, 
                            {weight_name: measure_symmetric_kl_divergence(
                                parameters_to_ndarrays(fit_in.parameters), 
                                parameters_to_ndarrays(fit_in_neighbor_client.parameters))
                            }
                        )
                    )

        #create graph for visual and search
        client_graph.add_edges_from(kl_list) 
        return client_graph
    
    def get_shortest_path_nodes_edges_weight_labels_subgraphs(
        client_graph,
        shortest_paths, 
        client_cids,
        weight='kl_div'
    ):
        shortest_path_nodes_edges_weight_labels_subgraphs = list()
        for cid in client_cids:
            total_shortest_paths = len(shortest_paths[cid])
            for i, path in enumerate(range(total_shortest_paths)): #
                path_edges = list(zip(shortest_paths[cid][path], shortest_paths[cid][path][1:]))
                path_edges = [(edge[0], edge[1], edge[2]) 
                            for edge in client_graph.edges(data=True) 
                            for path_edge in path_edges 
                            if edge[0]==path_edge[0] and edge[1]==path_edge[1] or edge[0]==path_edge[1] and edge[1]==path_edge[0]]
                path_nodes = [(i[0], i[1]) for i in path_edges]
                temp_graph = nx.Graph()
                temp_graph.add_edges_from(path_edges)
                weight_labels = dict([[(edge[0], edge[1]), edge[2][weight]] for edge in path_edges])
                shortest_path_nodes_edges_weight_labels_subgraphs.append(
                    {
                        'cid': cid,
                        'path_id': i,
                        'path_edges': path_edges,
                        'path_nodes': path_nodes,
                        'graph': json_graph.node_link_data(temp_graph, edges='links'),
                        'weight_labels': weight_labels
                    }
                )
        return shortest_path_nodes_edges_weight_labels_subgraphs

    def get_selected_shortest_paths(
        client_graph, client_cids, divergence_threshold=0.5, 
        weight='kl_div', method="dijkstra"
    ):
        shortest_paths = defaultdict()
        for cid in client_cids:
            shortest_paths_current_client = [
                nx.shortest_path(G=client_graph, source=str(cid), target=str(_cid), weight=weight, method=method) 
                for _cid in client_cids
                if cid!=_cid
            ]
            shortest_paths[cid] = shortest_paths_current_client
            
        selected_shortest_paths = defaultdict()
        for cid in client_cids:
            shortest_path_lengths = np.array([len(path) for path in shortest_paths[cid]])
            shortest_path_length_ratios = (shortest_path_lengths/shortest_path_lengths.sum()).round(1)
            selected_idx = int(
                random.choice(
                    np.where(
                        shortest_path_length_ratios==min(shortest_path_length_ratios, key=lambda x: abs(x - divergence_threshold))
                    )[0]
                )
            )
            selected_shortest_paths[cid] = [shortest_paths[cid][selected_idx]]
        
        return shortest_paths, selected_shortest_paths

    def get_client_specific_pairs(
        shortest_path_nodes_edges_weight_labels_subgraphs,
        client_cids
    ):
        client_specific_pairs = defaultdict(list)
        for cid in client_cids:
            for shortest_path_nodes_edges_weight_labels_subgraph in shortest_path_nodes_edges_weight_labels_subgraphs:  
                if cid==shortest_path_nodes_edges_weight_labels_subgraph['cid']:
                    client_specific_pairs[cid].append(shortest_path_nodes_edges_weight_labels_subgraph['path_edges'])
        return client_specific_pairs
    
    ##add KL aggregate and aggregate_and_return_fitins
    def kl_aggregate(current_client_idx, weights, all_fit_res):
        """
        1. take all incoming weights
        2. add a weight of 1.0 for the current client to weights
        3. normalize all weights 0-1
        4. aggregate params for all clients using normalized weights
        5. take weighted average of the current client
        6. add 4 and 5
        """
        max_weight = np.float64(np.array([i[1] for i in weights]).max()) #pull the max weight and assign to the current client - the current client and most closest client will have the highest weight
        current_client_cid_and_weight = (all_fit_res[current_client_idx][0].cid, max_weight) #current client is assigned the highest weight 
        weights.append(current_client_cid_and_weight) #append weight for current client to all incoming weights
        print(f"Weights for the current round before normalization: {weights}")
        
        weights = weights[::-1] #reverse list
        all_weights_sum = sum([i[1] for i in weights]) #sum all weights to get total_sum for normalization - remember weight for the client is at index 1
        final_adjusted_weights = [(i[0], i[1]/all_weights_sum) for i in weights] #normalie all weights between 0-1 
        current_client_cid_and_weight_normalized = final_adjusted_weights[0] #get the weight for the current client
        print(f"Weights for the current round after normalization: {final_adjusted_weights}")
        
        # print(f"current client cid and weights adjusted: {current_client_cid_and_weight_normalized}")
        # client_fit_res_as_key_value = defaultdict()
        params = parameters_to_ndarrays(all_fit_res[current_client_idx][1].parameters) # fit_res is a list of tuples of [client_proxy, client_fit_res]
        params = [p*current_client_cid_and_weight_normalized[1] for p in params] #take weighted average of the current client
        # print(f"final adjusted weights: {final_adjusted_weights}")
        all_non_current_client_params = list()
        for i, (client_proxy, fit_res) in enumerate(all_fit_res):
            for client_weight in final_adjusted_weights[1:]: #loop over all client except for the current client which is at index 0
                # print(f"Client proxy weight inside loop: {client_proxy.cid} | Client weight inside loop: {client_weight}")
                if client_proxy.cid == client_weight[0]:
                    # print(f"Client cid: {client_weight[0]} | Client weight: {client_weight[1]}")
                    non_current_client_params = [
                        p * client_weight[1] 
                        for p in parameters_to_ndarrays(fit_res.parameters) # client_weight[0] is the name of the current client given as cid
                    ]
                    all_non_current_client_params.append(non_current_client_params)
        non_current_client_params = [np.sum(arrays, axis=0) for arrays in zip(*all_non_current_client_params)]
        aggregated_params = [current_client_param + non_current_client_param for current_client_param, non_current_client_param in zip(params, non_current_client_params)]
        return aggregated_params
    
    def aggregate_and_return_fitins(all_fit_res, client_specific_pairs, config={}, weight_name='kl_div'):
        aggregated_fit_ins = []
        print(f"All client specific pairs in aggregate and return fitins: {client_specific_pairs}")
        for current_client_idx, (current_client_proxy, _) in enumerate(all_fit_res):
            # Pull the connected clients by the current client's cid - there is a slight difference between mst and sp since you are indexing on [0] because path_edges are already in a list. 
            # Pull the inner list before looping over it.
            current_client_specific_pairs = client_specific_pairs[current_client_proxy.cid][0] 
            print(f"Current client specific pairs in aggregate and return fitins: {current_client_specific_pairs} | current client proxy cid: {current_client_proxy.cid}")
            #pull weights from the pairs which show a connection to the current client
            client_weights = list()
            for pair in current_client_specific_pairs:
                client_weights.append(pair[2][weight_name])
            
            # print(f"Client weights: {client_weights}")
            #inverse and normalize weights - taking the inverse means that client with less KL divergence has more weight 
            client_weights = 1/(np.array(client_weights)+1e-6) #add a small constant to avoid divide by zero
            normalized_client_weights = client_weights/client_weights.sum()
            client_weights = list()
            for pair, w in zip(current_client_specific_pairs, normalized_client_weights):
                if pair[0]!=current_client_proxy.cid:
                    client_weights.append((pair[0], w))
                elif pair[1]!=current_client_proxy.cid:
                    client_weights.append((pair[1], w))
            
            current_client_agg_params = kl_aggregate(
                current_client_idx, 
                client_weights, 
                all_fit_res=all_fit_res
            )
            config["cid"] = current_client_proxy.cid
            config = dict(config)
            # print(f"Inside aggregate_and_return_fitins: Current Client Proxy CID: {current_client_proxy.cid}| Current client config dict: {config} | Len client agg param: {len(current_client_agg_params)}")
            aggregated_fit_ins.append(
                (
                    current_client_proxy.cid,
                    FitIns(
                        parameters=ndarrays_to_parameters(current_client_agg_params),
                        config=config
                    )
                )
            )
        return aggregated_fit_ins
    
    def aggregate_drift_sp(all_fit_res, server_round, weight='kl_div', method='kruskal', divergence_threshold=0.5, config={}):
        
        # get all client cids
        client_cids = [c.cid for c, _ in all_fit_res]
        
        # generate client graph
        client_graph = get_client_graph(
            all_fit_res=all_fit_res,
            weight_name=weight
        )
        
        # get all shortest paths between all clients and selected shortest paths based on divergence threshold
        shortest_paths, selected_shortest_paths = get_selected_shortest_paths(
            client_graph=client_graph, 
            client_cids=client_cids, 
            divergence_threshold=divergence_threshold,
            weight=weight,
            method=method,
        )
        
        # save shortest path and selected shortest paths with server round
        graph_dict = {
            "server_round": server_round,
            "client_graph": json_graph.node_link_data(client_graph, link="links"),
            "model_name": friendly_model_name,
            "shortest_paths": shortest_paths,
            "selected_shortest_paths": selected_shortest_paths,
            "divergence_threshold": divergence_threshold,
            "random_state": seed
        }
        append_graph_dict_to_json(
            filename='data/drift/drift_sp_graphs.json',
            graph_data=graph_dict
        )

        # get all path nodes, path edges, weight labels, subgraphs for the selected shortest paths
        shortest_path_nodes_edges_weight_labels_subgraphs = get_shortest_path_nodes_edges_weight_labels_subgraphs(
            client_graph=client_graph,
            shortest_paths=selected_shortest_paths, 
            client_cids=client_cids
        )

        # get client specific pairs for aggregation
        client_specific_pairs = get_client_specific_pairs(
            shortest_path_nodes_edges_weight_labels_subgraphs, 
            client_cids
        )
        print(client_specific_pairs)
        
        # aggregated clients based on client specific pairs
        aggregated_fit_ins = aggregate_and_return_fitins(
            all_fit_res=all_fit_res,
            client_specific_pairs=client_specific_pairs,
            config=config
        )
        
        return aggregated_fit_ins
    
    class DriftSPModelStrategy(fl.server.strategy.FedAvg):
        
        def configure_fit(self, server_round, parameters, client_manager):
            #if its round 1 then only initial parameters are sent to the clients
            if server_round==1:
                return super().configure_fit(server_round, parameters, client_manager)
            if server_round>1:
                config = defaultdict()
                config["server_round"] = server_round 
                config["strategy"]= strategy_name
                
                # clients which did not participate in the previous round get the standard aggregated parameters using fedavg strategy
                # Sample clients
                sample_size, min_num_clients = self.num_fit_clients(
                    client_manager.num_available()
                )
                # print(sample_size, min_num_clients)
                clients = client_manager.sample(
                    num_clients=sample_size, min_num_clients=min_num_clients
                )
                print(f"Config before drift aggregation: {config}")
                aggregated_fit_ins = aggregate_drift_sp(
                    all_fit_res=self.all_fit_res,
                    server_round=server_round,
                    weight=weight,
                    method=method,
                    divergence_threshold=divergence_threshold,
                    config=config
                )

                # Return client/config pairs
                configure_client_fitins = []
                for client in clients:
                    for fit_ins in aggregated_fit_ins:
                        if int(client.cid)==int(fit_ins[0]):
                            configure_client_fitins.append((client, FitIns(fit_ins[1].parameters, {"server_round": server_round, "strategy": strategy_name, "cid": client.cid})))#fit_ins[1]))
                
                print(f"Len of configure client fitins: {len(configure_client_fitins)}")
                for client, fit_ins in configure_client_fitins:
                    print(f"Sending - Sampled Client CID: {client.cid} | Fit in config: {fit_ins.config}")
        
                print("*"*100)
                
                existing_clients = [int(c.cid) for c, _ in configure_client_fitins] #list of all the clients which have aggregated parameters - all the others get fedavg aggregated params
                for client in clients:
                    if int(client.cid) not in existing_clients:
                        configure_client_fitins.append((client, FitIns(parameters, {"server_round": server_round, "strategy": strategy_name, "cid": client.cid})))
                    
                
                print(f"Len of updated configure client fitins: {len(configure_client_fitins)}")
                for client, fit_ins in configure_client_fitins:
                    print(f"Sending - Sampled Client CID: {client.cid} | Fit in config: {fit_ins.config}")
        
                print("*"*100)
                
                return configure_client_fitins
                
                    
        def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            """Aggregate model weights using weighted average and store checkpoint"""
            # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics - returns aggregated parameters and aggregated_metrics
            self.all_fit_res = results # these are all FitRes which will be used on configure_fit
            
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

    strategy = DriftSPModelStrategy(
        fraction_fit = fraction_fit,
        fraction_evaluate = 0,
        min_fit_clients = int(fraction_fit*client_config_file.NUM_CLIENTS),
        min_evaluate_clients = 0,
        min_available_clients = client_config_file.NUM_CLIENTS,
        on_fit_config_fn = client_config,
        on_evaluate_config_fn= client_config,
        initial_parameters=initial_parameters
    )
    return strategy
