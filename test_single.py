import torch
import os
import numpy as np
from argparse import ArgumentParser
from transformers import AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training


from model.gpt2classifier import GPT2Classifier
from model.bertclassifier import BERTClassifier
from model.llama2classifier import Llama2Classifier

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, TOKENIZERS_PARALLELISM=False)
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    if args.model_name == "meta-llama/Llama-2-7b-hf":
        base_model = Llama2Classifier
    elif args.model_name == "bert-base-uncased":
        base_model = BERTClassifier
    
    if args.quantize_for_lora:
        model = base_model(class_num=args.class_num, max_length=args.max_length, quantize_for_lora=args.quantize_for_lora)
    else:
        model = base_model(class_num=args.class_num, max_length=args.max_length) 
    model.model.resize_token_embeddings(len(tokenizer))

    if args.quantize:
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        print("Using Quantized Model")
        
    if args.quantize_for_lora:
        model = prepare_model_for_kbit_training(model)
        print("Using Quantized Model for LoRA")       

    if args.lora:
        if args.model_name == "meta-llama/Llama-2-7b-hf":
            lora_config = LoraConfig(r=16, lora_alpha=64, lora_dropout=0.1, bias="none",
                                    task_type=TaskType.SEQ_CLS,
                                    target_modules=['v_proj', 'down_proj', 'up_proj', 'q_proj', 'gate_proj', 'k_proj', 'o_proj'])
        elif args.model_name == "bert-base-uncased":
            lora_config = LoraConfig(r=16, lora_alpha=64, lora_dropout=0.1, bias="none",
                                    task_type=TaskType.SEQ_CLS,
                                    target_modules=['query', 'key', 'value', 'dense'])
        else:
            raise Exception("LoRA Config for the given model is not defined yet!")
        model = get_peft_model(model, lora_config)

    if args.balance:
        args.checkpoint_dir = f"{args.checkpoint_dir}_bal"
    if args.proportional:
        args.checkpoint_dir = f"{args.checkpoint_dir}_prop"
    if args.quantize:
        args.checkpoint_dir = f"{args.checkpoint_dir}_quant"
    if args.quantize_for_lora:
        args.checkpoint_dir = f"{args.checkpoint_dir}_quant4L"
    if args.lora:
        args.checkpoint_dir = f"{args.checkpoint_dir}_lora"
        
    model.load_state_dict(torch.load(os.path.join(f"{args.checkpoint_dir}-{args.model_name}", "model.pt")))
    if not args.cpu:
        model.cuda()
    model.eval()

    
    if args.input == None:
        args.input = """
        """
                
    if args.instruction:
        instruction = """## Instruction: Classify the [MBTI] of the given [Text].
        [Text]
        know intj tool use interaction people excuse antisocial truly enlighten mastermind know would count pet peeze something time matter people either whether group people mall never see best friend sit outside conversation jsut listen want interject sit formulate say wait inject argument thought find fascinate sit watch people talk people fascinate sit class watch different people find intrigue dad intj u stand look like line safeway watch people home talk people like think military job people voluntarily go job important show deference endanger live glorify way civilian think pretty ignorant general think military necessary defense mechanism political tactic feel like u specifically invest much money could put money education whatnot though personally sound budget aernative really comment one way base two politician eye year ago come name somewhat important kinda role model nowadays pick keep score individual level mean little vary accord number condition day may score high others low sweat really good cast physiotherapist like fiberglass cast break arm whatever sometimes want take picture beast put someone arm sadly people blind brilliance need tell directly wave arm frantically totally beyond oblivious get good eye contact help lot start find like attention get opposite sex notice however gay men tend little aggressive always walk away flatter like alcohol bad start generally keep go pas run money even mention fact crave cocaine drink political power mainly desire form power okay status still never study day life never learn study feel like real whatever reason college prepare recieve people like depend career introductory course help start rid bike write essay etc choose career least stimulate mind expand perspective reality without college like kiss sound ear yup roll end quite strange confession time mind wish people le judgemental self perceive flaw run situation person confess something expect judge one way another freak realize
        [MBTI]
        intj
        [Text]
        """
        model_input = f"{instruction}{' '.join(args.input.split(' ')[:args.max_length//2])}\n[MBTI]\n"
    else:
        model_input = f"{' '.join(args.input.split(' ')[:args.max_length-10])}"
        
    input_ids = tokenizer(model_input,
                          return_tensors="pt",
                          padding="longest",
                          max_length = args.max_length,
                          truncation=True).input_ids
    model_device = next(model.parameters()).device
    model_output = model(input_ids=input_ids.to(model_device)).cpu()
    
    print(f"Model output: {model_output}")

    model_answer_idx = torch.argmax(model_output, dim=1).numpy()
    model_top_answers_idx = torch.argsort(model_output, dim=1, descending=True).numpy()

    class_names = np.array(["intj", "intp", "infj", "infp",
                            "istj", "istp", "isfj", "isfp",
                            "entj", "entp", "enfj", "enfp",
                            "estj", "estp", "esfj", "esfp"])

    model_answer = class_names[model_answer_idx[0]]
    model_top_answers = class_names[model_top_answers_idx[0]]

    print(f"Model Answer: {model_answer}")
    print(f"Top Answers: {' '.join(model_top_answers)}")
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--checkpoint_dir", type=str, default="model_checkpoints/MBTI500")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--class_num", type=int, default=16)
    parser.add_argument("--balance", type=bool, default=False)
    parser.add_argument("--proportional", type=bool, default=False)
    parser.add_argument("--instruction", type=bool, default=False)
    parser.add_argument("--quantize", type=bool, default=False)
    parser.add_argument("--quantize_for_lora", type=bool, default=False)
    parser.add_argument("--cpu", type=bool, default=False)
    parser.add_argument("--lora", type=bool, default=False)
    parser.add_argument("--input", type=str, default=None)
    args = parser.parse_args() 
    main(args)