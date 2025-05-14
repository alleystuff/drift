import copy
import torch
from config_folder import client_config_file
from collections import defaultdict, OrderedDict

def set_torch_device(manual_seed=42, strategy="FedAvg"):
    """
    Set device for mps, cuda
    Set manual seed
    """
    if torch.backends.mps.is_available():
        device = client_config_file.DEVICE_MPS
        torch.mps.manual_seed(manual_seed)
    elif torch.cuda.is_available():
        device_count = [i for i in range(torch.cuda.device_count())]
        if strategy=="FedAvg":#expand to all strategies and use multiple processes to parallelize
            device = f"cuda:{device_count[0]}"
        elif strategy=="FedAvgM":#expand to all strategies and use multiple processes to parallelize
            device = f"cuda:{device_count[1]}"
        elif strategy=="FedProx":#expand to all strategies and use multiple processes to parallelize
            device = f"cuda:{device_count[2]}"
        else:#expand to all strategies and use multiple processes to parallelize
            device = f"cuda:{device_count[0]}"
        device = 'cuda'
        torch.cuda.manual_seed(manual_seed)
    else:
        device = client_config_file.DEVICE_CPU
        torch.manual_seed(manual_seed)

    print(f"Device: {device} | Manual Seed set to {manual_seed}")
    return device

def notebook_line_magic():
    """
    Avoid having to restart kernel when working with python scripts
    """
    from IPython import get_ipython
    ip = get_ipython()
    ip.run_line_magic("reload_ext", "autoreload")
    ip.run_line_magic("autoreload", "2")
    print("Line Magic Set")
    
def append_dict_to_json(file_path, new_data):
    """Appends a dictionary to a JSON file.

    Args:
        file_path (str): The path to the JSON file.
        new_data (dict): The dictionary to append.
    """
    import json
    try:
        with open(file_path, 'r') as file:
            file_data = json.load(file)
            if isinstance(file_data, list):
                file_data.append(new_data)
            with open(file_path, 'w') as file:
                json.dump(file_data, file, indent=4)
    except FileNotFoundError:
        with open(file_path, 'w') as file:
            json.dump([new_data], file, indent=4)
    except json.JSONDecodeError:
        print(f"Json decode error while writing thought data to file. File path: {file_path}")
    except ValueError as e:
        print(f"ValueError: {e}")