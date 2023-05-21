import os
import pickle
import numpy as np
import torch
from torch import nn
from transformers import AutoModel
from transformers import AutoTokenizer

# Set environment variables
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load data and model paths
labeled_df_fullcomments, labeled_df = pickle.load(open('/path/to/your/network_setup_topics_tuples.pkl', 'rb'))
model_path = "/path/to/your/categorize_33topics_weights.pt"
encoding_path = "/path/to/your/encoding_dict.pkl"

device = 'cuda:0'
threshold = 0.5

# Load encoding dictionary and ensure consistent order
encode_reverse = pickle.load(open(encoding_path, 'rb'))
encode_reverse = np.array(list(encode_reverse.values()))

class MyBert(nn.Module):
    """Custom BERT model for multi-label classification."""
    
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

    def forward(self,
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
                return_dict=None):
        s1 = self.pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask,
                             inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, past_key_values=past_key_values,
                             use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        downs_topics = self.multilabel_layers(s1['pooler_output'])

        if output_hidden_states:
            return s1['hidden_states']
        elif output_attentions:
            return s1['attentions']
        elif output_hidden_states and output_attentions:
            return s1['hidden_states'], s1['attentions']
        else:
            return downs_topics

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModel
import torch
import gc
import numpy as np
import pandas as pd
import pickle

class CustomDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['comment_sent']
        text_len = self.tokenizer(text, truncation=True, max_length=512)
        text_len = sum(text_len['attention_mask'])
        pt_batch = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        tmp = pt_batch['input_ids'].clone()
        pt_batch['input_ids'] = tmp.squeeze()
        tmp = pt_batch['attention_mask'].clone()
        pt_batch['attention_mask'] = tmp.squeeze()
        return pt_batch, torch.tensor(text_len)

    def __len__(self):
        return len(self.df)

# Replace the following paths with your own paths
labeled_df_path = "/path/to/your/labeled_df.pkl"
output_embeddings_path = "/path/to/your/output_embeddings.pkl"

labeled_df = pickle.load(open(labeled_df_path, "rb"))

tokenizer1 = AutoTokenizer.from_pretrained("xlm-roberta-base")
xlmr_dataset = CustomDataset(labeled_df, tokenizer1)
dataloader1 = DataLoader(
    xlmr_dataset, batch_size=32, num_workers=16, shuffle=False
)

tokenizer2 = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base")
xlmt_dataset = CustomDataset(labeled_df, tokenizer2)
dataloader2 = DataLoader(
    xlmt_dataset, batch_size=32, num_workers=16, shuffle=False
)

tokenizer3 = AutoTokenizer.from_pretrained("microsoft/infoxlm-base")
infoxlm_dataset = CustomDataset(labeled_df, tokenizer3)
dataloader3 = DataLoader(
    infoxlm_dataset, batch_size=32, num_workers=16, shuffle=False
)

tokenizer4 = AutoTokenizer.from_pretrained("microsoft/xlm-align-base")
xlmalign_dataset = CustomDataset(labeled_df, tokenizer4)
dataloader4 = DataLoader(
    xlmalign_dataset, batch_size=32, num_workers=16, shuffle=False
)

dataloader_li = [dataloader1, dataloader2, dataloader3, dataloader4]
device_li = ['cuda:0', 'cuda:1', 'cuda:1', 'cuda:0']

# Replace the following path with your own model path
model_path = "/path/to/your/model.pth"

model1 = mybert()
loaded_state_dict = torch.load(model_path, map_location='cuda:0')
model1.load_state_dict(loaded_state_dict)

config = AutoConfig.from_pretrained("cardiffnlp/twitter-xlm-roberta-base")
model2 = AutoModel.from_pretrained("cardiffnlp/twitter-xlm-roberta-base", config=config)

config = AutoConfig.from_pretrained("microsoft/infoxlm-base")
model3 = AutoModel.from_pretrained("microsoft/infoxlm-base", config=config)

config = AutoConfig.from_pretrained("microsoft/xlm-align-base")
model4 = AutoModel.from_pretrained("microsoft/xlm-align-base", config=config)

model_li = [model1, model2, model3, model4]
weight_li = [4, 0.5, 1, 1.5]

# Define the extract_features_emb function here

embeddings = extract_features_emb(dataloader_li, model_li, device_li, weight_li)

labeled_df['Embeddings'] = embeddings.tolist()
pickle.dump(obj=labeled_df, file=open(output_embeddings_path, 'wb'))

def calc_scores(vectors, labels):
    """
    Calculate various clustering scores for the given vectors and labels.

    Args:
        vectors (numpy.ndarray): The feature vectors.
        labels (numpy.ndarray): The cluster labels.

    Returns:
        tuple: A tuple containing silhouette score, Davies-Bouldin score,
               Calinski-Harabasz score, and other scores.
    """
    idx_take = np.argwhere(labels != -1).squeeze()
    stat = stats.mode(idx_take)

    try:
        noise_len = len(np.argwhere(labels == -1).squeeze())
    except TypeError:
        noise_len = 0

    try:
        most_label, most_count = stat.mode[0], stat.count[0]
        count_penalty = most_count / (len(labels) - noise_len)
    except IndexError:
        most_count = 0
        count_penalty = 0

    noise_score = noise_len / len(labels)
    eval_labels = labels[idx_take]
    eval_vec = vectors[idx_take, :]

    try:
        S_score = metrics.silhouette_score(eval_vec, eval_labels, metric='cosine')
        D_score = metrics.davies_bouldin_score(eval_vec, eval_labels)
        C_score = metrics.calinski_harabasz_score(eval_vec, eval_labels)
    except ValueError:
        S_score = 0
        D_score = 0
        C_score = 0

    try:
        hdbscan_score_c, hdbscan_score_e = hdbscan_scorer(eval_vec, eval_labels)
    except timeout_decorator.TimeoutError:
        hdbscan_score_c = 0
        hdbscan_score_e = 0

    try:
        hdbscan_score2_c, hdbscan_score2_e = hdbscan_scorer2(vectors, labels)
    except timeout_decorator.TimeoutError:
        hdbscan_score2_c = 0
        hdbscan_score2_e = 0

    return (S_score, D_score, C_score, hdbscan_score_c, hdbscan_score_e,
            hdbscan_score2_c, hdbscan_score2_e, noise_score, count_penalty)

# Replace the following paths with your own paths
comment_table_path = "/path/to/your/comment_table.csv"
cluster_table_path = "/path/to/your/cluster_table.csv"
topic_statistic_path = "/path/to/your/topic_statistic.csv"

comment_table.to_csv(comment_table_path, index=False)
cluster_table.to_csv(cluster_table_path, index=False)
topic_statistic_df.to_csv(topic_statistic_path, index=False)