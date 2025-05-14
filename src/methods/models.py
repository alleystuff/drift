import os
os.environ['HF_HOME'] = "drift/hub" #on gpu 0

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import gc
import re
from collections import defaultdict
from utils.training_utils import get_quantization_config
from config_folder.model_config import MODELS

class StateModel:
    def __init__(self, model_name, gpu_id=None, quantization="4bit"):
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.quantization=quantization
        self.model_loaded = False #flag to make sure that the model is not loaded multiple times
        
    def set_tokenizer(self):
        if self.model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        else:
            return "Specify model name."
    
    def load_model(self):
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        self.set_tokenizer() #set the tokenizer
        # if self.model_loaded==False: #turn on the flag if the model is already loaded
        if torch.cuda.is_available():
            device_map = f"cuda:{self.gpu_id}" if self.gpu_id!=None else "auto"
            print(f"Device map set to: {device_map}")
        else:
            device_map = "cpu"
            quantization_config = None
            
        quantization_config = get_quantization_config(quantization=self.quantization)
        print(f"Loading {self.model_name}| Device map {device_map} | Quantization: {self.quantization}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device_map,         # Map the entire model to the specified GPU
            torch_dtype=torch.float16,       # Optionally use mixed precision
            quantization_config=quantization_config,  # Pass the BitsAndBytesConfig
        )
            
        self.model_loaded=True #turn on the flag if the model is already loaded
        
    def generate(
        self, 
        prompt, 
        **parameters
    ):
        self.load_model() 
        try:
            max_new_tokens = parameters['parameters']["max_new_tokens"] if "max_new_tokens" in parameters['parameters'] else 50
            num_return_sequences = parameters['parameters']["num_return_sequences"] if "num_return_sequences" in parameters['parameters'] else 1
            temperature = parameters['parameters']["temperature"] if "temperature" in parameters['parameters'] else 0.8
            top_p = parameters['parameters']["top_p"] if "top_p" in parameters['parameters'] else 0.99
            do_sample = parameters['parameters']["do_sample"] if "topdo_sample_p" in parameters['parameters'] else True
            repetition_penalty = parameters['parameters']["repetition_penalty"] if "repetition_penalty" in parameters['parameters'] else 1.1
        except:
            parameters = MODELS[self.model_name]
            max_new_tokens = parameters['parameters']["max_new_tokens"] if "max_new_tokens" in parameters['parameters'] else 50
            num_return_sequences = parameters['parameters']["num_return_sequences"] if "num_return_sequences" in parameters['parameters'] else 1
            temperature = parameters['parameters']["temperature"] if "temperature" in parameters['parameters'] else 0.8
            top_p = parameters['parameters']["top_p"] if "top_p" in parameters['parameters'] else 0.99
            do_sample = parameters['parameters']["do_sample"] if "topdo_sample_p" in parameters['parameters'] else True
            repetition_penalty = parameters['parameters']["repetition_penalty"] if "repetition_penalty" in parameters['parameters'] else 1.1
            
        # print(f"Max new tokens: {max_new_tokens} | Num return sequences: {num_return_sequences} | Temperature: {temperature}")
        if self.model_name=="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" or self.model_name=="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B":
            messages = [
                {"role": "user", "content": prompt}
            ]
            # Apply chat template
            self.tokenizer.pad_token_id = self.model.config.pad_token_id #self.tokenizer.eos_token_id
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize and generate
            inputs = self.tokenizer(text, return_tensors="pt")
            input_token_len = inputs['input_ids'].shape[-1] #size of the input tokens
            # print(f"Deepseek input token len: {input_token_len}")
            #put tokens on device
            if torch.cuda.is_available() and self.gpu_id!=None:
                inputs = inputs.to(f"cuda:{self.gpu_id}") 
            elif torch.cuda.is_available() and self.gpu_id==None:
                inputs = inputs.to(f"cuda")
            else:
                inputs = inputs.to("cpu")
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,  # Maximum number of tokens in the output
                # max_length=50,
                num_return_sequences=num_return_sequences,  # Number of sequences to generate
                temperature=temperature,  # Adjust temperature for randomness (lower is more deterministic)
                top_p=top_p,  # Nucleus sampling (top-p sampling)
                do_sample=do_sample,  # Enable sampling
                repetition_penalty=repetition_penalty,
                use_cache=True,
                pad_token_id=self.model.config.pad_token_id
            )

            response = self.tokenizer.decode(
                outputs[0][input_token_len:], 
                skip_special_tokens=True
            )
            
            try:
                response = response.split("</think>")[1]
            except:
                response = response
            
            try:
                response = re.findall(r'\d+', response)
                response = response[-1]
                if int(response)>=1 and int(response)<=10:
                    return response
            except:
                return None
        elif self.model_name=="meta-llama/Llama-2-13b-chat-hf":
            self.tokenizer.pad_token_id = self.model.config.pad_token_id #self.tokenizer.eos_token_id
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", #pt is for pytorch
                add_special_tokens=True,
                # padding=True
            )
            # attention_mask = inputs["attention_mask"]
            input_token_len = inputs['input_ids'].shape[-1] #size of the input tokens
            
            #put tokens on device
            if torch.cuda.is_available() and self.gpu_id!=None:
                inputs = inputs.to(f"cuda:{self.gpu_id}") 
            elif torch.cuda.is_available() and self.gpu_id==None:
                inputs = inputs.to(f"cuda")
            else:
                inputs = inputs.to("cpu")
              
            output_ids = self.model.generate(
                **inputs,
                # attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,  # Maximum number of tokens in the output
                num_return_sequences=num_return_sequences,  # Number of sequences to generate
                temperature=temperature,  # Adjust temperature for randomness (lower is more deterministic)
                top_p=top_p,  # Nucleus sampling (top-p sampling)
                do_sample=do_sample,  # Enable sampling
                repetition_penalty=repetition_penalty,
            )
            output_text = self.tokenizer.decode(
                output_ids[0][input_token_len:], 
                skip_special_tokens=True
            )
            try:
                response = output_text.split("System: ")[1] if "System: " in output_text else output_text.split(":")[1]
                return response
            except IndexError:
                return output_text
        elif self.model_name=="Qwen/Qwen2.5-7B-Instruct":
            messages = [
                {"role": "user", "content": prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer([text], return_tensors="pt")#.to(model.device)
            #put tokens on device
            if torch.cuda.is_available() and self.gpu_id!=None:
                inputs = inputs.to(f"cuda:{self.gpu_id}") 
            elif torch.cuda.is_available() and self.gpu_id==None:
                inputs = inputs.to(f"cuda")
            else:
                inputs = inputs.to("cpu")

            generated_ids = self.model.generate(
                **inputs,
                # attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,  # Maximum number of tokens in the output
                num_return_sequences=num_return_sequences,  # Number of sequences to generate
                temperature=temperature,  # Adjust temperature for randomness (lower is more deterministic)
                top_p=top_p,  # Nucleus sampling (top-p sampling)
                do_sample=do_sample,  # Enable sampling
                repetition_penalty=repetition_penalty,
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response
        else:
            messages = [
                # {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            # self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
            self.tokenizer.pad_token_id = self.model.config.pad_token_id #self.tokenizer.eos_token_id
            text = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            inputs = self.tokenizer(text, return_tensors="pt")

            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            
            #put tokens on device
            if torch.cuda.is_available() and self.gpu_id!=None:
                inputs = inputs.to(f"cuda:{self.gpu_id}") 
            elif torch.cuda.is_available() and self.gpu_id==None:
                inputs = inputs.to(f"cuda")
            else:
                inputs = inputs.to("cpu")
                
            output_ids = self.model.generate(
                **inputs,
                # attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,  # Maximum number of tokens in the output
                num_return_sequences=num_return_sequences,  # Number of sequences to generate
                temperature=temperature,  # Adjust temperature for randomness (lower is more deterministic)
                top_p=top_p,  # Nucleus sampling (top-p sampling)
                do_sample=do_sample,  # Enable sampling
                repetition_penalty=repetition_penalty,
                eos_token_id=terminators,
                pad_token_id=self.tokenizer.eos_token_id
            )
            output_text = self.tokenizer.decode(
                output_ids[0][inputs['input_ids'].shape[-1]:], 
                skip_special_tokens=True
            )
            try:
                # response = output_text.split("System: ")[1] if "System: " in output_text else output_text.split(":")[1]
                return output_text #response
            except IndexError:
                return output_text

    def empty_cache(self):
        torch.cuda.empty_cache()
        self.model_loaded=False

class ServerModel:
    def __init__(self, model_name, gpu_id=None, quantization="4bit"):
        gc.collect()
        torch.cuda.empty_cache()
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.quantization=quantization
        if self.model_name==None:
            return "Specify model name."
        self.device_map = f"cuda:{self.gpu_id}" if self.gpu_id!=None and type(self.gpu_id)==int else "auto"
        print(f"Device map set to: {device_map}")
        if self.gpu_id=="cpu":
            device_map = "cpu"        
        quantization_config = get_quantization_config(quantization=self.quantization)
        print(f"Loading {self.model_name}| Device map {device_map} | Quantization: {self.quantization}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device_map,         # Map the entire model to the specified GPU
            torch_dtype=torch.float16,       # Optionally use mixed precision
            quantization_config=quantization_config,  # Pass the BitsAndBytesConfig
        )
        
    def generate(
        self, 
        prompt, 
        **parameters):
        try:
            max_new_tokens = parameters['parameters']["max_new_tokens"] if "max_new_tokens" in parameters['parameters'] else 50
            num_return_sequences = parameters['parameters']["num_return_sequences"] if "num_return_sequences" in parameters['parameters'] else 1
            temperature = parameters['parameters']["temperature"] if "temperature" in parameters['parameters'] else 0.8
            top_p = parameters['parameters']["top_p"] if "top_p" in parameters['parameters'] else 0.99
            do_sample = parameters['parameters']["do_sample"] if "topdo_sample_p" in parameters['parameters'] else True
            repetition_penalty = parameters['parameters']["repetition_penalty"] if "repetition_penalty" in parameters['parameters'] else 1.1
        except:
            parameters = MODELS[self.model_name]
            max_new_tokens = parameters['parameters']["max_new_tokens"] if "max_new_tokens" in parameters['parameters'] else 50
            num_return_sequences = parameters['parameters']["num_return_sequences"] if "num_return_sequences" in parameters['parameters'] else 1
            temperature = parameters['parameters']["temperature"] if "temperature" in parameters['parameters'] else 0.8
            top_p = parameters['parameters']["top_p"] if "top_p" in parameters['parameters'] else 0.99
            do_sample = parameters['parameters']["do_sample"] if "topdo_sample_p" in parameters['parameters'] else True
            repetition_penalty = parameters['parameters']["repetition_penalty"] if "repetition_penalty" in parameters['parameters'] else 1.1
            
        # print(f"Max new tokens: {max_new_tokens} | Num return sequences: {num_return_sequences} | Temperature: {temperature}")
        if self.model_name=="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" or self.model_name=="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B":
            messages = [
                {"role": "user", "content": prompt}
            ]
            # Apply chat template
            self.tokenizer.pad_token_id = self.model.config.pad_token_id #self.tokenizer.eos_token_id
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize and generate
            inputs = self.tokenizer(text, return_tensors="pt")
            input_token_len = inputs['input_ids'].shape[-1] #size of the input tokens
            # print(f"Deepseek input token len: {input_token_len}")
            #put tokens on device
            inputs = inputs.to(self.device_map)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,  # Maximum number of tokens in the output
                # max_length=50,
                num_return_sequences=num_return_sequences,  # Number of sequences to generate
                temperature=temperature,  # Adjust temperature for randomness (lower is more deterministic)
                top_p=top_p,  # Nucleus sampling (top-p sampling)
                do_sample=do_sample,  # Enable sampling
                repetition_penalty=repetition_penalty,
                use_cache=True,
                pad_token_id=self.model.config.pad_token_id
            )

            response = self.tokenizer.decode(
                outputs[0][input_token_len:], 
                skip_special_tokens=True
            )
            
            try:
                response = response.split("</think>")[1]
            except:
                response = response
            
            try:
                response = re.findall(r'\d+', response)
                response = response[-1]
                if int(response)>=1 and int(response)<=10:
                    return response
            except:
                return None
        elif self.model_name=="meta-llama/Llama-2-13b-chat-hf":
            self.tokenizer.pad_token_id = self.model.config.pad_token_id #self.tokenizer.eos_token_id
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", #pt is for pytorch
                add_special_tokens=True,
                # padding=True
            )
            # attention_mask = inputs["attention_mask"]
            input_token_len = inputs['input_ids'].shape[-1] #size of the input tokens
            
            #put tokens on device
            inputs = inputs.to(self.device_map)
              
            output_ids = self.model.generate(
                **inputs,
                # attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,  # Maximum number of tokens in the output
                num_return_sequences=num_return_sequences,  # Number of sequences to generate
                temperature=temperature,  # Adjust temperature for randomness (lower is more deterministic)
                top_p=top_p,  # Nucleus sampling (top-p sampling)
                do_sample=do_sample,  # Enable sampling
                repetition_penalty=repetition_penalty,
            )
            output_text = self.tokenizer.decode(
                output_ids[0][input_token_len:], 
                skip_special_tokens=True
            )
            try:
                response = output_text.split("System: ")[1] if "System: " in output_text else output_text.split(":")[1]
                return response
            except IndexError:
                return output_text
        elif self.model_name=="Qwen/Qwen2.5-7B-Instruct":
            messages = [
                {"role": "user", "content": prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer([text], return_tensors="pt")#.to(model.device)
            #put tokens on device
            inputs = inputs.to(self.device_map)

            generated_ids = self.model.generate(
                **inputs,
                # attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,  # Maximum number of tokens in the output
                num_return_sequences=num_return_sequences,  # Number of sequences to generate
                temperature=temperature,  # Adjust temperature for randomness (lower is more deterministic)
                top_p=top_p,  # Nucleus sampling (top-p sampling)
                do_sample=do_sample,  # Enable sampling
                repetition_penalty=repetition_penalty,
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response
        else:
            messages = [
                # {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            # self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
            self.tokenizer.pad_token_id = self.model.config.pad_token_id #self.tokenizer.eos_token_id
            text = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            inputs = self.tokenizer(text, return_tensors="pt")
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            #put tokens on device
            inputs = inputs.to(self.device_map)
            output_ids = self.model.generate(
                **inputs,
                # attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,  # Maximum number of tokens in the output
                num_return_sequences=num_return_sequences,  # Number of sequences to generate
                temperature=temperature,  # Adjust temperature for randomness (lower is more deterministic)
                top_p=top_p,  # Nucleus sampling (top-p sampling)
                do_sample=do_sample,  # Enable sampling
                repetition_penalty=repetition_penalty,
                eos_token_id=terminators,
                pad_token_id=self.tokenizer.eos_token_id
            )
            output_text = self.tokenizer.decode(
                output_ids[0][inputs['input_ids'].shape[-1]:], 
                skip_special_tokens=True
            )
            try:
                # response = output_text.split("System: ")[1] if "System: " in output_text else output_text.split(":")[1]
                return output_text #response
            except IndexError:
                return output_text

    def empty_cache(self):
        torch.cuda.empty_cache()
        self.model_loaded=False
          
class TrainedModel:
    """Class to be used with trained models
    """
    def __init__(self, base_model_name, saved_model_path=None, tokenizer_path=None, model=None, tokenizer=None, gpu_id=None, quantization="4bit"):
        from dotenv import load_dotenv
        load_dotenv()
        
        if gpu_id==None:
            gpu_id="auto"
        else:
            gpu_id = f"cuda:{gpu_id}"
        
        self.gpu_id = gpu_id
        if model!=None and tokenizer!=None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            print(f"Using a trained model from the following save path: {saved_model_path} on device {self.gpu_id}")
            quantization_config = get_quantization_config(quantization=quantization)
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map=self.gpu_id,         # Map the entire model to the specified GPU
                torch_dtype=torch.float16,       # Optionally use mixed precision
                quantization_config=quantization_config,  # Pass the BitsAndBytesConfig

            )
            self.model = PeftModel.from_pretrained(self.base_model, saved_model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
    def generate(self, prompt="sup", **parameters):
        # print("max_new_tokens" in parameters['parameters'])
        max_new_tokens = parameters['parameters']["max_new_tokens"] if "max_new_tokens" in parameters['parameters'] else 50
        num_return_sequences = parameters['parameters']["num_return_sequences"] if "num_return_sequences" in parameters['parameters'] else 1
        temperature = parameters['parameters']["temperature"] if "temperature" in parameters['parameters'] else 0.8
        top_p = parameters['parameters']["top_p"] if "top_p" in parameters['parameters'] else 0.99
        do_sample = parameters['parameters']["do_sample"] if "topdo_sample_p" in parameters['parameters'] else True
        repetition_penalty = parameters['parameters']["repetition_penalty"] if "repetition_penalty" in parameters['parameters'] else 1.1
        
        messages = [
            {"role": "user", "content": prompt},
        ]
        
        self.tokenizer.pad_token_id = self.model.config.pad_token_id #self.tokenizer.eos_token_id
        text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        # inputs_gpu = "cuda:1" 
        inputs_gpu = f"cuda:{torch.cuda.current_device()}"
        inputs = self.tokenizer(text, return_tensors="pt").to(inputs_gpu)

        
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        terminators = [i for i in terminators if type(i)==int]
        # print(inputs)
        # print(max_new_tokens, num_return_sequences, temperature, top_p, do_sample, repetition_penalty, terminators)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,  # Maximum number of tokens in the output
            num_return_sequences=num_return_sequences,  # Number of sequences to generate
            temperature=temperature,  # Adjust temperature for randomness (lower is more deterministic)
            top_p=top_p,  # Nucleus sampling (top-p sampling)
            do_sample=do_sample,  # Enable sampling
            repetition_penalty=repetition_penalty,
            eos_token_id=terminators,
            pad_token_id=self.tokenizer.eos_token_id
        )
        output_text = self.tokenizer.decode(
            output_ids[0][inputs['input_ids'].shape[-1]:], 
            skip_special_tokens=True
        )
        try:
            # response = output_text.split("System: ")[1] if "System: " in output_text else output_text.split(":")[1]
            return output_text #response
        except IndexError:
            return output_text
