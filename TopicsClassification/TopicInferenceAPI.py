import json
import os
import pickle

import numpy as np
import torch
import transformers
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

# Set environment variables
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load parameters from JSON file
with open('path/to/your/parameter.json', 'r') as f:
    param_dict = json.load(f)

model_path = param_dict['model_path']
encoding_path = param_dict['encoding_path']
device = param_dict['gpu_device']
threshold = param_dict['threshold']

encode_reverse = pickle.load(open(encoding_path, 'rb'))
encode_reverse = np.array(list(encode_reverse.values()))

class MyBert(nn.Module):
    def __init__(self):
        super(MyBert, self).__init__()
        self.pretrained = AutoModel.from_pretrained('xlm-roberta-base')
        self.multilabel_layers = nn.Sequential(nn.Linear(768, 256),
                                               nn.Mish(),
                                               nn.BatchNorm1d(256),
                                               nn.Dropout(0.1),
                                               nn.Linear(256, 64),
                                               nn.Mish(),
                                               nn.BatchNorm1d(64),
                                               nn.Dropout(0.1),
                                               nn.Linear(64, len(encode_reverse))
                                           )

    def forward(self, **kwargs):
        s1 = self.pretrained(**kwargs)
        downs_topics = self.multilabel_layers(s1['pooler_output'])

        if kwargs.get('output_hidden_states', False):
            return s1['hidden_states']
        elif kwargs.get('output_attentions', False):
            return s1['attentions']
        elif kwargs.get('output_hidden_states', False) and kwargs.get('output_attentions', False):
            return s1['hidden_states'], s1['attentions']
        else:
            return downs_topics

category_model = MyBert()

# Load model weights
loaded_state_dict = torch.load(model_path, map_location=device)
category_model.load_state_dict(loaded_state_dict)

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

sig_func = nn.Sigmoid().to(device)
category_model.to(device).eval()

def analyze_text_topics_inhouse(text):
    inputs = tokenizer(
            [text],
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors="pt",
            )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        pred_topics = category_model(**inputs)
    pred_topics_score = sig_func(pred_topics).detach().cpu().numpy()
    pred_topics = np.where(pred_topics_score > threshold, 1, 0)
    idxli = np.argwhere(pred_topics == 1)[:, 1]
    topics = list(encode_reverse[idxli])
    score_list = list(pred_topics_score[0][idxli])
    topics_text = []
    topics_text.append(dict(
        text=text,
        topics_list=topics,
        score_list=str(score_list)
    ))
    json_object = json.dumps(topics_text)
    print(json_object)
    return json_object

text = "I dont care, it's a bad product. I don't want to use it anymore"
analyze_text_topics_inhouse(text)