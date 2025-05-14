import copy
import torch
from collections import OrderedDict

def measure_kl_divergence(params_A, params_B):
    import numpy as np
    from scipy.special import kl_div

    kl_divergence = 0
    for a, b in zip(params_A, params_B):
        temp_masked = np.ma.masked_values(kl_div(a.reshape(-1), b.reshape(-1)), np.inf)
        temp_mean = temp_masked[~temp_masked.mask].data.mean()
        kl_divergence += temp_mean
    
    return np.round(np.mean(kl_divergence), 4)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def get_peft_parameters_and_peft_state_dict(peft_model, adpater_name="default"):
    """
    Retruns:
    - parameters: list of numpy arrays containing adapter weights
    - peft_state_dict: state dict containining keys(layer names) and values(weights) of the adapter
    """
    from peft import get_peft_model_state_dict
    # set_peft_model_state_dict,
    # get_eva_state_dict #for SVD
    
    print_trainable_parameters(peft_model)
    
    params_dict = copy.deepcopy(
        OrderedDict((name, param.detach()) for name, param in peft_model.named_parameters() if
                    adpater_name in name))

    peft_state_dict = get_peft_model_state_dict(
        model=peft_model,   
        state_dict=params_dict,
        adapter_name=adpater_name
    )
    parameters = [i.cpu().numpy() for i in list(peft_state_dict.values())]
    
    count = 0
    for i in parameters:
        count+=torch.numel(torch.from_numpy(i))
    
    print(f"Total trainable parameters from the Peft Model: {count}")
    
    return parameters, peft_state_dict

def set_peft_state_dict(peft_model, peft_state_dict, adapter_name="baseline"):
    """
    Set the state dict of the peft model with the given peft_state_dict.
    This will merge the adapter back in.
    return:
    - peft_model
    """
    from peft import set_peft_model_state_dict
    
    model = set_peft_model_state_dict(
        model=peft_model,
        peft_model_state_dict=peft_state_dict,
        adapter_name=adapter_name
    )
    return model

def get_quantization_config(quantization="4bit"):
    from transformers import BitsAndBytesConfig
    if quantization=="4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,  # or "float16"
            bnb_4bit_use_double_quant=True,  # Use double quantization for lower memory usage
            # llm_int8_enable_fp32_cpu_offload=True  # Ensure CPU offload is properly set
        )
        return quantization_config
    if quantization=="8bit":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        return quantization_config