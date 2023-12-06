import os
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig


class Llama2Classifier(nn.Module):
    def __init__(self, class_num=16, max_length=512, quantize_for_lora=False, model_config=None):
        super().__init__()
        if quantize_for_lora:
            q_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-2-7b-hf", 
                                                                            quantization_config=q_config, 
                                                                            num_labels=class_num, 
                                                                            max_position_embeddings=max_length)
        else:
            if model_config is None:
                self.model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-2-7b-hf",
                                                                                num_labels=class_num,
                                                                                max_position_embeddings=max_length)
            else:
                self.model = AutoModelForSequenceClassification.from_config(model_config,
                                                                            num_labels=class_num,
                                                                            max_position_embeddings=max_length)

    
    def forward(self, input_ids, attention_mask=None):
        model_outputs = self.model(input_ids = input_ids,
                                   attention_mask = attention_mask)
        return F.softmax(model_outputs.logits, dim=1)
        