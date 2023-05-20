import numpy as np
import transformers
import torch
from torch import nn
import os
import pickle
import json
from transformers import AutoConfig, AutoTokenizer, AutoModel

# Set environment variables
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

# Load parameters from JSON file
with open('./parameter.json','r') as f:
    param_dict = json.load(f)
model_path = param_dict['model_path'] # Path to the model
encoding_path = param_dict['encoding_path'] # Path to the encoding
xlmr_model = param_dict['bert_model_path'] # Path to the BERT model
xlmr_tokenizer = param_dict['tokenizer_path'] # Path to the tokenizer
device = param_dict['gpu_device'] # GPU device to use
threshold = param_dict['threshold'] # Threshold for topic classification

# Load encoding
encode_reverse = pickle.load(open(encoding_path,'rb'))
encode_reverse = np.array(list(encode_reverse.values()))

# Define customized model
class mybert(nn.Module):
    def __init__(self):
        super(mybert,self).__init__()
        self.pretrained = AutoModel.from_pretrained(xlmr_model)
        self.multilabel_layers = nn.Sequential(nn.Linear(768, 256),
                                               nn.Mish(),
                                               nn.BatchNorm1d(256),
                                               nn.Dropout(0.1),
                                               nn.Linear(256, 64),
                                               nn.Mish(),
                                               nn.BatchNorm1d(64),
                                               nn.Dropout(0.1),
                                               nn.Linear(64,len(encode_reverse))
                                           )
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        s1 = self.pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, 
                            inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, past_key_values=past_key_values, 
                            use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        downs_topics = self.multilabel_layers(s1['pooler_output'])
        
        if output_hidden_states==True:
            return s1['hidden_states']
        elif output_attentions==True:
            return s1['attentions']
        elif (output_hidden_states==True) and (output_attentions==True):
            return s1['hidden_states'],s1['attentions']
        else:
            return  downs_topics

# Load model and tokenizer
category_model = mybert()
loaded_state_dict = torch.load(model_path,map_location=device)
category_model.load_state_dict(loaded_state_dict)
tokenizer = AutoTokenizer.from_pretrained(xlmr_tokenizer)

# Define sigmoid function and set model to evaluation mode
sig_func = nn.Sigmoid().to(device)
category_model.to(device).eval()

# Define function to analyze text topics
def analyze_text_topics_inhouse(text):
    raw_inputs = [text]
    inputs = tokenizer(
            raw_inputs, 
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors="pt",
            )
    inputs = {k: v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
        pred_topics  = category_model(**inputs)
    pred_topics_score = sig_func(pred_topics).detach().cpu().numpy()
    pred_topics = np.where(pred_topics_score>threshold,1,0)
    idxli = np.argwhere(pred_topics==1)[:,1]
    topics = list(encode_reverse[idxli])
    score_list = list(pred_topics_score[0][idxli])    
    topics_text = []
    topics_text.append(dict(
        text=text,
        topics_list=topics,
        score_list = str(score_list)
    ))
    json_object = json.dumps(topics_text)
    print(json_object)   
    return json_object

# Demo data:
# text = "Make the devices able to be setup without requiring an internet connection"
# analyze_text_topics_inhouse(text)