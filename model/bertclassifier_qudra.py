import os
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig


class BERTClassifierQudra(nn.Module):
    def __init__(self, class_num=2, max_length=512, quantize_for_lora=False, model_config=None):
        super().__init__()
        if quantize_for_lora:
            raise NotImplementedError("quantize_for_lora is not implemented for BERTClassifierQudra")
        else:
            self.classifiers = nn.ModuleList()
            for i in range(4): 
                if model_config is None:
                    self.classifiers.append(AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",
                                                                                               num_labels=class_num,
                                                                                               max_position_embeddings=max_length))
                else:
                    self.classifiers.append(AutoModelForSequenceClassification.from_config(model_config,
                                                                                           num_labels=class_num,
                                                                                           max_position_embeddings=max_length))

    
    def forward(self, input_ids, attention_mask=None):
        model_outputs = [classifier(input_ids = input_ids,
                                    attention_mask = attention_mask) for classifier in self.classifiers]
        model_probs = [F.softmax(model_output.logits, dim=1) for model_output in model_outputs]
        
        return torch.stack(model_probs, dim=1)
        