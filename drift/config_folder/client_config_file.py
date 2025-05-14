DEVICE_MPS = "mps"
DEVICE_CUDA = "cuda"
DEVICE_CPU = "cpu"

MODEL_NAMES = ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]
STRATEGIES = ['FedAvg', 'FedProx', 'Drift', 'FedAdam', 'FedTrimmedAvg']
TEST_RANDOM_STATES=[40, 42, 48]
NUM_CLIENTS=8
LOCAL_TRAINING_STEPS=70
DATASETS = {
    "csqa": {
        "name": "tau/commonsense_qa",
        "splits": ["train", "validation", "test"]
    },
    "cose": {
        "name": "Salesforce/cos_e",
        "splits": ["train", "validation"]    
    },
    "piqa": {
        "name":"ybisk/piqa",
        "splits":["train", "validation", "test"]    
    },
    "aqua": {
        "name":"deepmind/aqua_rat",
        "splits":["train", "validation", "test"]
    },
    "mathqa": {
        "name":"allenai/math_qa",
        "splits":["train", "validation", "test"]
    },
    "medqa": {
        "name":"bigbio/med_qa",
        "splits": ["train", "validation", "test"]
    },
    "medmcqa": {
        "name":"openlifescienceai/medmcqa",
        "splits": ["train", "validation", "test"]
    },
    "pubmedqa": {
        "name":"qiaojin/PubMedQA",
        "splits":["train"]
    },
}

#server checkpoint path
def get_server_checkpoint_path(filename, model_type="baseline"):
    """_summary_

    Args:
        filename (_type_): base path
        type (str, optional): one of baseline, clients, or server. Defaults to "baseline".

    Returns:
        path: path to the saved model checkpoint
    """
    return f"src/saved_models/{model_type}/{filename}.pth"

def get_client_checkpoint_path(filename):
    return f"src/saved_models/custom_clients/{filename}.pth"