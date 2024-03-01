import torch
import os

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from argparse import ArgumentParser

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, TOKENIZERS_PARALLELISM=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': ["[Text]", "[MBTI]",
                                                            "intj", "intp", "infj", "infp",
                                                            "istj", "istp", "isfj", "isfp",
                                                            "entj", "entp", "enfj", "enfp",
                                                            "estj", "estp", "esfj", "esfp"]})
    
    model_config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_config(model_config)
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(os.path.join(f"{args.checkpoint_dir}-{args.model_name}", "model.pt")))
    model = model.to(torch.device("cuda:0"))
    model.eval()
    
    instruction = """## Instruction: Classify the [MBTI] of the given [Text].
    [Text]
    know intj tool use interaction people excuse antisocial truly enlighten mastermind know would count pet peeze something time matter people either whether group people mall never see best friend sit outside conversation jsut listen want interject sit formulate say wait inject argument thought find fascinate sit watch people talk people fascinate sit class watch different people find intrigue dad intj u stand look like line safeway watch people home talk people like think military job people voluntarily go job important show deference endanger live glorify way civilian think pretty ignorant general think military necessary defense mechanism political tactic feel like u specifically invest much money could put money education whatnot though personally sound budget aernative really comment one way base two politician eye year ago come name somewhat important kinda role model nowadays pick keep score individual level mean little vary accord number condition day may score high others low sweat really good cast physiotherapist like fiberglass cast break arm whatever sometimes want take picture beast put someone arm sadly people blind brilliance need tell directly wave arm frantically totally beyond oblivious get good eye contact help lot start find like attention get opposite sex notice however gay men tend little aggressive always walk away flatter like alcohol bad start generally keep go pas run money even mention fact crave cocaine drink political power mainly desire form power okay status still never study day life never learn study feel like real whatever reason college prepare recieve people like depend career introductory course help start rid bike write essay etc choose career least stimulate mind expand perspective reality without college like kiss sound ear yup roll end quite strange confession time mind wish people le judgemental self perceive flaw run situation person confess something expect judge one way another freak realize
    [MBTI]
    intj
    [Text]
    """
    args.input = """I wish to make known to the Russian people, to Papa [Nicholas II] the Russian mother and to the children, to the land of Russia, what they must understand. If I am killed by common assassins, and especially by my brothers the Russian peasants, you, Tsar of Russia, have nothing to fear, remain on your throne and govern... But if I am murdered by nobles and if they shed my blood, their hands will remain soiled with my blood, for twenty-five years they will not wash their hands from my blood. They will leave Russia. Brothers will kill brothers... if it was your relations who have wrought my death then no one of your family, that is to say none of your children or relations, will remain alive for more then two years."""
    model_input = f"{instruction}{' '.join(args.input.split(' ')[:300])}\n[MBTI]\n"
    input_ids = tokenizer(model_input,
                          return_tensors="pt",
                          padding="longest",
                          max_length = args.max_length,
                          truncation=True).input_ids
    naive_answer = tokenizer.decode(torch.argmax(model(input_ids=input_ids.to(model.device)).logits[0][-1]))
    print(f"Model Answer: {naive_answer}")
    
    top_tokens = torch.argsort(model(input_ids=input_ids.to(model.device)).logits[0][-1], descending=True)
    top_answers = " ".join([tokenizer.decode(top_tokens[i]) for i in range(10)])
    print(f"Top Answers: {top_answers}")
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--checkpoint_dir", type=str, default="model_checkpoints/MBTI500")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--input", type=str, default="")
    args = parser.parse_args() 
    main(args)