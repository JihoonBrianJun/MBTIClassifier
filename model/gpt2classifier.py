from torch import nn
from torch.nn import functional as F
from transformers import GPT2Model


class GPT2Classifier(nn.Module):
    def __init__(self, class_num=16, model_config=None):
        super().__init__()
        if model_config is None:
            self.model = GPT2Model.from_pretrained("gpt2")
        else:
            self.model = GPT2Model(model_config)
            
        self.classifier_head = nn.Linear(self.model.config.hidden_size, class_num, bias=False)
    
    def forward(self, input_ids, attention_mask=None):
        model_outputs = self.model(input_ids = input_ids,
                                   attention_mask = attention_mask)
        class_logits = self.classifier_head(model_outputs.last_hidden_state[:,-1,:])
        return F.softmax(class_logits, dim=1)
        