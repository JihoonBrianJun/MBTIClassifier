import torch
import os
from argparse import ArgumentParser
from transformers import AutoTokenizer
from torch.utils.mobile_optimizer import optimize_for_mobile

from model.bertclassifier import BERTClassifier

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, TOKENIZERS_PARALLELISM=False)
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    dummy_input = tokenizer("Hi "*(args.max_length-2), return_tensors="pt").input_ids

    base_model = BERTClassifier
    model = base_model(class_num=args.class_num, max_length=args.max_length) 
    model.model.resize_token_embeddings(len(tokenizer))

    if args.balance:
        args.checkpoint_dir = f"{args.checkpoint_dir}_bal_quant"
    elif args.proportional:
        args.checkpoint_dir = f"{args.checkpoint_dir}_prop_quant"
    else:
        args.checkpoint_dir = f"{args.checkpoint_dir}_quant"

    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    model.load_state_dict(torch.load(os.path.join(f"{args.checkpoint_dir}-{args.model_name}", "model.pt")))
    model.eval()

    traced_model = torch.jit.trace(model, dummy_input)
    optimized_model = optimize_for_mobile(traced_model)
    
    save_dir = f"save/{args.model_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    optimized_model._save_for_lite_interpreter(os.path.join(save_dir, "bertMBTIClassifier.ptl"))
    tokenizer.save_vocabulary(save_dir)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--checkpoint_dir", type=str, default="model_checkpoints/MBTI500")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--class_num", type=int, default=16)
    parser.add_argument("--balance", type=bool, default=False)
    parser.add_argument("--proportional", type=bool, default=False)
    args = parser.parse_args() 
    main(args)