
import numpy as np
import transformers
import torch
from torch import nn
import os
import pickle
import json

from transformers import AutoConfig, AutoTokenizer, AutoModel

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load parameters from a JSON file
with open('./parameter.json', 'r') as f:
    param_dict = json.load(f)

model_path = param_dict['model_path']  # Path to the model
encoding_path = param_dict['encoding_path']  # Path to the encoding
device = param_dict['gpu_device']  # GPU device to use
threshold = param_dict['threshold']  # Threshold for topic classification

# Load the encoding
encode_reverse = pickle.load(open(encoding_path, 'rb'))
encode_reverse = np.array(list(encode_reverse.values()))  # Ensure the order is consistent

# Define a customized model
class MyBERT(nn.Module):
    def __init__(self):
        super(MyBERT, self).__init__()
        self.pretrained = AutoModel.from_pretrained('xlm-roberta-base')
        self.multilabel_layers = nn.Sequential(
            nn.Linear(768, 256),
            nn.Mish(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.Mish(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.Linear(64, len(encode_reverse))  # Change to the number of subtech labels
        )
        
    def forward(self, **inputs):
        s1 = self.pretrained(**inputs)
        downs_topics = self.multilabel_layers(s1['pooler_output'])
        
        if inputs.get('output_hidden_states', False):
            return s1['hidden_states']
        elif inputs.get('output_attentions', False):
            return s1['attentions']
        elif inputs.get('output_hidden_states', False) and inputs.get('output_attentions', False):
            return s1['hidden_states'], s1['attentions']
        else:
            return downs_topics

category_model = MyBERT()

# Load the model state dictionary
loaded_state_dict = torch.load(model_path, map_location=device)
category_model.load_state_dict(loaded_state_dict)

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

sig_func = nn.Sigmoid().to(device)

category_model.to(device).eval()

def analyze_text_topics_inhouse(text):
    raw_inputs = [text]
    inputs = tokenizer(
        raw_inputs, 
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
    topics_text.append({
        'text': text,
        'topics_list': topics,
        'score_list': str(score_list)
    })
    
    json_object = json.dumps(topics_text)
    print(json_object)
    
    return json_object

# Demo data
text = "I dont care, it's a bad product. I don't want to use it anymore"
analyze_text_topics_inhouse(text)