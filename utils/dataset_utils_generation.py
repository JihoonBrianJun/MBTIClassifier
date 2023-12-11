import copy
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, Sequence
import pandas as pd
import torch
from torch.utils.data import Dataset

class DatasetGenerator(Dataset):
    def __init__(self, tokenizer, config, dataset_type):
        super(DatasetGenerator, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.IGNORE_INDEX = -100
        
        if dataset_type == "train":
            data_path = config.train_data_path
        else:
            data_path = config.eval_data_path
        
        instruction = """## Instruction: Classify the [MBTI] of the given [Text].
        [Text]
        know intj tool use interaction people excuse antisocial truly enlighten mastermind know would count pet peeze something time matter people either whether group people mall never see best friend sit outside conversation jsut listen want interject sit formulate say wait inject argument thought find fascinate sit watch people talk people fascinate sit class watch different people find intrigue dad intj u stand look like line safeway watch people home talk people like think military job people voluntarily go job important show deference endanger live glorify way civilian think pretty ignorant general think military necessary defense mechanism political tactic feel like u specifically invest much money could put money education whatnot though personally sound budget aernative really comment one way base two politician eye year ago come name somewhat important kinda role model nowadays pick keep score individual level mean little vary accord number condition day may score high others low sweat really good cast physiotherapist like fiberglass cast break arm whatever sometimes want take picture beast put someone arm sadly people blind brilliance need tell directly wave arm frantically totally beyond oblivious get good eye contact help lot start find like attention get opposite sex notice however gay men tend little aggressive always walk away flatter like alcohol bad start generally keep go pas run money even mention fact crave cocaine drink political power mainly desire form power okay status still never study day life never learn study feel like real whatever reason college prepare recieve people like depend career introductory course help start rid bike write essay etc choose career least stimulate mind expand perspective reality without college like kiss sound ear yup roll end quite strange confession time mind wish people le judgemental self perceive flaw run situation person confess something expect judge one way another freak realize
        [MBTI]
        intj
        [Text]
        """
        
        df = pd.read_csv(data_path)
        df["posts_with_instruction"] = df["posts"].apply(lambda x: f"{instruction}{' '.join(x.split(' ')[:300])}\n[MBTI]\n")
        df["MBTI"] = df["type"].apply(lambda x: x.lower())
        
        input_list = df["posts_with_instruction"].tolist()
        target_list = df["MBTI"].tolist()

        print("Tokenizing inputs... This may take some time...")
        if dataset_type == "train":
            example_list = [input+target for input, target in zip(input_list, target_list)]
        else:
            example_list = input_list
        
        example_tokenized, input_tokenized = [self.tokenize(data_list) 
                                            for data_list in (example_list, input_list)]
        input_ids = example_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        
        # instruction_tokenized_len = self.tokenize([instruction])["input_ids_lens"][0]
        
        print("Tokenizing All done!")
        
        for i in tqdm(range(len(labels))):
            input_len = input_tokenized["input_ids_lens"][i] 
            labels[i][:input_len] = self.IGNORE_INDEX
            # labels[i][:instruction_tokenized_len] = self.IGNORE_INDEX
        
        self.input_ids = input_ids
        self.labels = labels
        self.dataset_type = dataset_type
        if dataset_type == "eval":
            self.gold_answer = target_list
        print(f"Example dataset..\nInput: {input_list[0]}\ninput_ids: {self.input_ids[0]}\nlabels: {self.labels[0]}")
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if self.dataset_type == "train":
            return dict(input_ids=self.input_ids[i], labels=self.labels[i], dataset_type=self.dataset_type, gold_answer = None)
        else:
            return dict(input_ids=self.input_ids[i], labels=self.labels[i], dataset_type=self.dataset_type, gold_answer = self.gold_answer[i])
    
    def tokenize(self, data_list):
        tokenized_list = []
        for data in data_list:
            tokenized_list.append(self.tokenizer(data,
                                                 return_tensors="pt",
                                                 padding="longest",
                                                 max_length = self.config.max_length,
                                                 truncation=True))
        input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_lens = [tokenized.input_ids.ne(self.tokenizer.pad_token_id).sum().item()
                          for tokenized in tokenized_list]
        
        return dict(input_ids=input_ids, input_ids_lens=input_ids_lens)


@dataclass
class DataCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, dataset_type, gold_answers = tuple([instance[key] for instance in instances]
                                                              for key in ("input_ids", "labels", "dataset_type", "gold_answer"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True,
                                                 padding_value=self.tokenizer.pad_token_id)
        
        if dataset_type == "train":
            return dict(input_ids=input_ids, 
                        labels=labels,
                        attention_mask=input_ids.ne(self.tokenizer.pad_token_id))
        else:
            return dict(input_ids=input_ids, 
                        labels=labels,
                        attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
                        gold_answers=gold_answers)            