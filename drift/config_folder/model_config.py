MODELS = {
    "meta-llama/Llama-3.1-8B-Instruct" : { 
        "parameters" : {
            "max_new_tokens": 500,
            "num_return_sequences": 1,
            "temperature": 0.6,
            "top_p": 0.9,
            "do_sample": True,
            "repetition_penalty":1.3
        },
        "model_name": "llama_3_1_8B_instruct"
    },
    "Qwen/Qwen2.5-7B-Instruct" : { 
        "parameters" : {
            "max_new_tokens": 500,
            "num_return_sequences": 1,
            "temperature": 0.6,
            "top_p": 0.9,
            "do_sample": True,
            "repetition_penalty":1.3
        },
        "model_name": "qwen_2_5_7B_instruct"
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" : { 
        "parameters" : {
            "max_new_tokens": 75,
            "num_return_sequences": 1,
            "temperature": 0.7,
            "top_p": 0.99,
            "do_sample": True,
            "repetition_penalty":1.2
        },
        "model_name": "deepseek32B"
    },
    # "meta-llama/Llama-2-13b-chat-hf" : { 
    #     "parameters" : {
    #         "max_new_tokens": 75,
    #         "num_return_sequences": 1,
    #         "temperature": 0.7,
    #         "top_p": 0.8,
    #         "do_sample": True,
    #         "repetition_penalty":1.2
    #     },
    #     "model_name": "llama_2_13B_chat_hf"
    # },
}

#use qwen as a second model