import os
from pathlib import Path
from argparse import ArgumentParser

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from transformers import AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from configs.train_config import TrainConfig
from configs.test_config import TestConfig
from utils.dataset_utils_qudra import MBTIDataset, MBTIDataCollator
from utils.train_utils_qudra import train
from utils.test_utils_qudra import test

from model.bertclassifier_qudra import BERTClassifierQudra


def get_checkpoint_save_dir(config):
    folder_name = (
    config.checkpoint_root_folder
    + "/"
    + config.checkpoint_folder
    + "-"
    + config.model_name
    )
    save_dir = Path.cwd() / folder_name

    return save_dir

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

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


def main(args):    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    if args.run_mode == "train":
        config = TrainConfig() 
    else:
        config = TestConfig()
    
    if args.model_name is not None:
        config.model_name = args.model_name

    Model = BERTClassifierQudra
    config.batch_size_training = 8
    config.micro_batch_size = 8
    
    config.class_num = 2
    config.checkpoint_folder = f"{config.checkpoint_folder}_qudra"
    
    if args.balance:
        config.train_data_path= "dataset/MBTI 500_train_bal.csv"
        config.eval_data_path= "dataset/MBTI 500_eval_bal.csv"
        config.checkpoint_folder = f"{config.checkpoint_folder}_bal"
    
    if not args.instruction:
        config.batch_size_training *= 2
        config.micro_batch_size *= 2

    # if args.lora:
    #     config.checkpoint_folder = f"{config.checkpoint_folder}_lora"

    torch.cuda.manual_seed(config.seed)
    torch.manual_seed(config.seed)
    
    gradient_accumulation_steps = config.batch_size_training // config.micro_batch_size
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, TOKENIZERS_PARALLELISM=False)
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
    if args.run_mode == "train":
        if args.instruction:
            dataset = MBTIDataset(tokenizer, config, dataset_type="train", put_instruction=True)
        else:
            dataset = MBTIDataset(tokenizer, config, dataset_type="train", put_instruction=False)
    else:
        if args.instruction:
            dataset = MBTIDataset(tokenizer, config, dataset_type="eval", put_instruction=True)
        else:
            dataset = MBTIDataset(tokenizer, config, dataset_type="eval", put_instruction=False)
    data_collator = MBTIDataCollator(tokenizer)
    
    # Create DataLoaders for the training and validation dataset
    if args.run_mode == "train":
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size_training,
            num_workers=config.num_workers_dataloader,
            drop_last=True,
            collate_fn=data_collator,
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.val_batch_size,
            num_workers=config.num_workers_dataloader,
            drop_last=True,
            collate_fn=data_collator,
        )

    checkpoint_dir = get_checkpoint_save_dir(config)
    
    if args.run_mode == "train":
        if args.quantize_for_lora:
            model = Model(class_num=config.class_num, max_length=config.max_length, quantize_for_lora=args.quantize_for_lora)
        else:
            model = Model(class_num=config.class_num, max_length=config.max_length)
        for i in range(4):
            model.classifiers[i].resize_token_embeddings(len(tokenizer))
        
        if args.load_full_ckpt:
            model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "model.pt")))

        if args.lora:
            if config.model_name == "meta-llama/Llama-2-7b-hf":
                lora_config = LoraConfig(r=16, lora_alpha=64, lora_dropout=0.1, bias="none",
                                        task_type=TaskType.SEQ_CLS,
                                        target_modules=['v_proj', 'down_proj', 'up_proj', 'q_proj', 'gate_proj', 'k_proj', 'o_proj'])
            elif config.model_name == "bert-base-uncased":
                lora_config = LoraConfig(r=16, lora_alpha=64, lora_dropout=0.1, bias="none",
                                        task_type=TaskType.SEQ_CLS,
                                        target_modules=['query', 'key', 'value', 'dense'])
            else:
                raise Exception("LoRA Config for the given model is not defined yet!")
            for i in range(4):
                model.classifiers[i] = get_peft_model(model.classifiers[i], lora_config)
            # if args.load_full_ckpt:
            #     model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "model.pt")))

            if args.load_full_ckpt:
                if config.model_name == "meta-llama/Llama-2-7b-hf":
                    for i in range(4):
                        model.classifiers[i].score.original_module.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"score.original_module{i}.pt")))
                elif config.model_name == "bert-base-uncased":
                    for i in range(4):
                        model.classifiers[i].classifier.original_module.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"classifier.original_module{i}.pt")))

        if args.quantize_for_lora:
            for i in range(4):
                model.classifiers[i].gradient_checkpointing_enable()
            model = prepare_model_for_kbit_training(model)
            print("Using Quantized Model for LoRA")

        if args.quantize:
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            print("Using Quantized Model")

        if not args.cpu:
            model.cuda()
        print_size_of_model(model)
        print_trainable_parameters(model)

        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.0)
        scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)

        if args.lora:
            config.checkpoint_folder = f"{config.checkpoint_folder}_lora"
        if args.quantize:
            config.checkpoint_folder = f"{config.checkpoint_folder}_quant"
        if args.quantize_for_lora:
            config.checkpoint_folder = f"{config.checkpoint_folder}_quant4L"
        checkpoint_dir = get_checkpoint_save_dir(config)
        
        train_results = train(
            model,
            dataloader,
            optimizer,
            scheduler,
            gradient_accumulation_steps,
            config,
            checkpoint_dir,
            args.lora
        )
    else:
        if args.quantize_for_lora:
            model = Model(class_num=config.class_num, max_length=config.max_length, quantize_for_lora=args.quantize_for_lora)
        else:
            model = Model(class_num=config.class_num, max_length=config.max_length) 
        for i in range(4):
            model.classifiers[i].resize_token_embeddings(len(tokenizer)) 

        if args.lora:
            if config.model_name == "meta-llama/Llama-2-7b-hf":
                lora_config = LoraConfig(r=16, lora_alpha=64, lora_dropout=0.1, bias="none",
                                        task_type=TaskType.SEQ_CLS,
                                        target_modules=['v_proj', 'down_proj', 'up_proj', 'q_proj', 'gate_proj', 'k_proj', 'o_proj'])
            elif config.model_name == "bert-base-uncased":
                lora_config = LoraConfig(r=16, lora_alpha=64, lora_dropout=0.1, bias="none",
                                        task_type=TaskType.SEQ_CLS,
                                        target_modules=['query', 'key', 'value', 'dense'])
            else:
                raise Exception("LoRA Config for the given model is not defined yet!")
            for i in range(4):
                model.classifiers[i] = get_peft_model(model.classifiers[i], lora_config)
        
        print("After LoRA")

        if args.quantize_for_lora:
            model = prepare_model_for_kbit_training(model)
            print("Using Quantized Model for LoRA")      

        if args.quantize:
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            print("Using Quantized Model")
        
        if args.lora:
            config.checkpoint_folder = f"{config.checkpoint_folder}_lora"
        if args.quantize:
            config.checkpoint_folder = f"{config.checkpoint_folder}_quant"
        if args.quantize_for_lora:
            config.checkpoint_folder = f"{config.checkpoint_folder}_quant4L"
        checkpoint_dir = get_checkpoint_save_dir(config)
        
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "model.pt")))

        if args.lora:
            if config.model_name == "meta-llama/Llama-2-7b-hf":
                for i in range(4):
                    model.classifiers[i].score.original_module.load_state_dict(torch.load(os.path.join(checkpoint_dir, "score.original_module.pt")))
            elif config.model_name == "bert-base-uncased":
                for i in range(4):
                    model.classifiers[i].classifier.original_module.load_state_dict(torch.load(os.path.join(checkpoint_dir, "classifier.original_module.pt")))

        if not args.cpu:
            model = model.cuda()
        print_size_of_model(model)
        print_trainable_parameters(model)
        
        test_results = test(model, dataloader)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_mode", type=str, choices=["train", "eval"], default="train")
    parser.add_argument("--instruction", type=bool, default=False)
    parser.add_argument("--balance", type=bool, default=False)
    parser.add_argument("--quantize", type=bool, default=False)
    parser.add_argument("--quantize_for_lora", type=bool, default=False)
    parser.add_argument("--cpu", type=bool, default=False)
    parser.add_argument("--lora", type=bool, default=False)
    parser.add_argument("--load_full_ckpt", type=bool, default=False)
    parser.add_argument("--model_name", type=str, default=None)
    args = parser.parse_args() 
    main(args)