from dataclasses import dataclass
from typing import Dict, Sequence
import pandas as pd
import torch
from torch.utils.data import Dataset

class MBTIDataset(Dataset):
    def __init__(self, tokenizer, config, dataset_type, put_instruction):
        super(MBTIDataset, self).__init__()
        
        self.config = config
        self.tokenizer = tokenizer

        self.class_names = ["intj", "intp", "infj", "infp",
                            "istj", "istp", "isfj", "isfp",
                            "entj", "entp", "enfj", "enfp",
                            "estj", "estp", "esfj", "esfp"]
        
        if dataset_type == "train":
            data_path = config.train_data_path
        else:
            data_path = config.eval_data_path

        
        df = pd.read_csv(data_path)
        df["MBTI"] = df["type"].apply(lambda x: x.lower())
        
        if put_instruction:
            instruction = """## Instruction: Classify the [MBTI] of the given [Text].
            [Text]
            know intj tool use interaction people excuse antisocial truly enlighten mastermind know would count pet peeze something time matter people either whether group people mall never see best friend sit outside conversation jsut listen want interject sit formulate say wait inject argument thought find fascinate sit watch people talk people fascinate sit class watch different people find intrigue dad intj u stand look like line safeway watch people home talk people like think military job people voluntarily go job important show deference endanger live glorify way civilian think pretty ignorant general think military necessary defense mechanism political tactic feel like u specifically invest much money could put money education whatnot though personally sound budget aernative really comment one way base two politician eye year ago come name somewhat important kinda role model nowadays pick keep score individual level mean little vary accord number condition day may score high others low sweat really good cast physiotherapist like fiberglass cast break arm whatever sometimes want take picture beast put someone arm sadly people blind brilliance need tell directly wave arm frantically totally beyond oblivious get good eye contact help lot start find like attention get opposite sex notice however gay men tend little aggressive always walk away flatter like alcohol bad start generally keep go pas run money even mention fact crave cocaine drink political power mainly desire form power okay status still never study day life never learn study feel like real whatever reason college prepare recieve people like depend career introductory course help start rid bike write essay etc choose career least stimulate mind expand perspective reality without college like kiss sound ear yup roll end quite strange confession time mind wish people le judgemental self perceive flaw run situation person confess something expect judge one way another freak realize
            [MBTI]
            intj
            [Text]
            """
            df["posts_for_input"] = df["posts"].apply(lambda x: f"{instruction}{' '.join(x.split(' ')[:config.max_length//2])}\n[MBTI]\n")
        else:
            df["posts_for_input"] = df["posts"].apply(lambda x: f"{' '.join(x.split(' ')[:config.max_length-10])}")
        
        input_list = df["posts_for_input"].tolist()
        print("Tokenizing inputs... This may take some time...")
        
        input_tokenized = self.tokenize(input_list)
        print("Tokenizing All done!")
        
        self.input_ids = input_tokenized
        self.target_labels = df["MBTI"].apply(lambda x: self.class_names.index(x)).tolist()
        self.dataset_type = dataset_type
        if dataset_type == "eval":
            self.gold_answer = df["MBTI"].tolist()
        print(f"Example dataset..\nInput: {input_list[0]}\ninput_ids: {self.input_ids[0]}\ntarget_labels: {self.target_labels[0]}")
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if self.dataset_type == "train":
            return dict(input_ids=self.input_ids[i], target_labels=self.target_labels[i], dataset_type=self.dataset_type, gold_answer = None)
        else:
            return dict(input_ids=self.input_ids[i], target_labels=self.target_labels[i], dataset_type=self.dataset_type, gold_answer = self.gold_answer[i])
    
    def tokenize(self, data_list):
        tokenized_list = []
        for data in data_list:
            tokenized_list.append(self.tokenizer(data,
                                                 return_tensors="pt",
                                                 padding="longest",
                                                 max_length = self.config.max_length,
                                                 truncation=True))
        input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
        
        return input_ids


@dataclass
class MBTIDataCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, target_labels, dataset_type, gold_answers= tuple([instance[key] for instance in instances]
                                                                    for key in ("input_ids", "target_labels", "dataset_type", "gold_answer"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)
        target_labels = torch.tensor(target_labels)
        
        if dataset_type == "train":
            return dict(input_ids=input_ids, 
                        target_labels=target_labels,
                        attention_mask=input_ids.ne(self.tokenizer.pad_token_id))
        else:
            return dict(input_ids=input_ids, 
                        target_labels=target_labels,
                        attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
                        gold_answers=gold_answers)            