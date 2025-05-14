import torch
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, PeftModel, PeftConfig

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7" 

def get_dpo_trainer(
    model, 
    tokenizer,
    train_dataset,
    test_dataset,
    batch_size=4,
    output_dir="src/saved_models/baseline",
    run_name="baseline",
    lora_rank=8,
    use_cpu=False,
    local_rank=0,
    device=1,
    fl_rounds=100
    ):
    """_summary_

    Args:
        model (_type_): main model
        tokenizer (_type_): tokenizer 
        preference_dataset (_type_): preference dataset
        batch_size (int, optional): batch size. Defaults to 4.
        output_dir (str, optional): where the model is saved. Defaults to "src/saved_models/baseline".
        run_name (str, optional): _description_. Defaults to "baseline".

    Returns:
        DPO_Trainer: dpo trainer to train the model
    """
    # if device and type(device)==int:
    # torch.cuda.set_device(7)
    # add fraction to account for the number of federated learning rounds
    if run_name=='federated':
        step_level = round(len(train_dataset)/(4*batch_size))//fl_rounds if round(len(train_dataset)/(4*batch_size))//fl_rounds > 0 else 1
        
    if run_name=='baseline':
        step_level = round(len(train_dataset)/(4*batch_size))
    
    #eval steps
    eval_steps = step_level//2 if step_level//2>0 else 1
    warmup_steps = step_level//4 if step_level//4>0 else 1
    logging_steps = step_level//3 if step_level//3>0 else 1
    print(f"Using step level for local training as: {step_level} | warmup: {warmup_steps} | eval steps: {eval_steps} | logging_steps: {logging_steps}")
    
    tokenizer.pad_token = tokenizer.eos_token
    training_args = DPOConfig(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=2,#batch_size,
        # max_steps=script_args.max_steps,
        max_steps=(step_level*2),
        logging_steps=logging_steps,
        # save_steps=script_args.save_steps,
        save_steps=(step_level*1),
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,
        learning_rate=5e-6,
        eval_strategy="steps",
        eval_steps=(step_level*2), #100
        output_dir=output_dir,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        optim="adamw_torch",
        bf16=True,
        remove_unused_columns=False,
        run_name=run_name,
        beta=0.2,
        max_length=1024,
        max_prompt_length=512,
        use_cpu=False,
        torch_empty_cache_steps=5,
        label_names=['chosen', 'rejected']
        # local_rank=-1,
    )
    # training_args.place_model_on_device=False
    
    # print(f"local rank: {local_rank}")

    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            # "out_proj",
            # "fc_in",
            # "fc_out",
            # "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # peft_model = PeftModel(
    #     model=model.model,
    #     peft_config=peft_config,
    #     adapter_name=run_name
    # ).to("cuda:0")
    
    # return peft_model
    return DPOTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    